"""Access-unit tracked broken-chain locator.

Given a *broken* sub-stream and the source stream it was cut from, find out
why an independent decoder freezes on it, prove the cause with minimal
what-if changes, shrink the stream to the shortest still-failing form, and
name a recovery position a player can actually seek to.

Everything is tracked per access unit (AU).  The sub-stream is segmented into
AUs from the bytes alone (``first_slice_segment_flag`` needs no parameter
sets), and each AU is aligned back to a source picture by a content digest of
its VCL NALs, so the source stream's resolved references (RPS) become the
dependency view of the sub-stream.  Two root causes are distinguished:

* ``missing_parameter_sets`` - the extractor failed to carry the active
  VPS/SPS/PPS, so decoding can only start where the stream happens to repeat
  them ("切入发生在参数集更新之后");
* ``missing_reference`` - a kept picture references an enhancement picture
  that was cropped away, so decoding dies at the first referring AU.

Two single-change experiments decide which one applies: *only* swap the
parameter sets, or *only* restore one reference.  Whichever change lets an
independent decoder recover is the root cause; the repair then applies just
that change (never silently pasting the whole source stream back), and the
first recovered frame is marked from the actual decoded pictures.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import av
import numpy as np

from .bitstream import (
    NAL_PPS, NAL_SPS, NAL_VPS, BitReader, StreamLayout, parse_stream,
    rbsp_from_ebsp, split_annex_b,
)
from .decode import STRICT_DECODER_OPTIONS, DecodedPicture, decode_reference
from .extractor import _parameter_sets

PARAM_TYPES = (NAL_VPS, NAL_SPS, NAL_PPS)
PARAM_NAMES = {NAL_VPS: "VPS", NAL_SPS: "SPS", NAL_PPS: "PPS"}
NAL_NAMES = {32: "VPS", 33: "SPS", 34: "PPS", 35: "AUD", 36: "EOS", 37: "EOB",
             39: "SEI", 40: "SEI_SUFFIX"}
_UNALIGNED_PTS = 1 << 40


class LocatorError(RuntimeError):
    """Raised when a locator precondition (e.g. a legal stream) is violated."""


class Cause(str, Enum):
    NONE = "none"
    MISSING_PARAMETER_SETS = "missing_parameter_sets"
    MISSING_REFERENCE = "missing_reference"
    UNKNOWN = "unknown"


def _nal_type(data: bytes) -> int:
    return (data[0] >> 1) & 0x3F


@dataclass
class AccessUnit:
    """One access unit of a sub-stream, aligned back to the source stream."""
    index: int
    prefix_nals: list[bytes]          # non-VCL payloads (AUD, parameter sets, SEI)
    vcl_nals: list[bytes]             # VCL payloads (one or more slices)
    source_index: int | None = None   # decode index in the source stream
    presentation: int | None = None
    temporal_id: int | None = None
    picture_type: str = "?"
    is_irap: bool = False
    refs: tuple[int, ...] = ()        # source decode indices this AU predicts from

    @property
    def parameter_types(self) -> list[int]:
        return [t for t in PARAM_TYPES
                if any(_nal_type(n) == t for n in self.prefix_nals)]

    def to_bytes(self) -> bytes:
        return b"".join(b"\x00\x00\x01" + n
                        for n in (*self.prefix_nals, *self.vcl_nals))


def split_access_units(blob: bytes) -> list[AccessUnit]:
    """Segment an Annex-B stream into access units without any parameter sets.

    A new AU starts at a VCL NAL whose ``first_slice_segment_flag`` is set;
    non-VCL NALs (AUD, parameter sets, SEI) attach to the AU that follows
    them, mirroring how the extractor prefixes pictures.
    """
    aus: list[AccessUnit] = []
    prefix: list[bytes] = []
    vcl: list[bytes] = []

    def flush() -> None:
        nonlocal prefix, vcl
        if vcl or prefix:
            aus.append(AccessUnit(len(aus), prefix, vcl))
        prefix, vcl = [], []

    for nu in split_annex_b(blob):
        if nu.is_vcl:
            first = False
            if len(nu.data) > 2:
                first = BitReader(rbsp_from_ebsp(nu.data[2:])).flag()
            if first and vcl:
                flush()
            vcl.append(nu.data)
        else:
            if vcl:
                flush()
            prefix.append(nu.data)
    flush()
    return aus


def _vcl_digest(vcl_nals: list[bytes]) -> str:
    return hashlib.sha1(b"".join(vcl_nals)).hexdigest()


def source_vcl_bytes(layout: StreamLayout, source_index: int) -> list[bytes]:
    p = layout.pictures[source_index]
    return [layout.nals[k].data
            for k in range(p.first_nal_index, p.last_nal_index + 1)]


def align_to_source(aus: list[AccessUnit], layout: StreamLayout) -> None:
    """Map each AU to its source picture by content digest of the VCL bytes.

    The extractor copies source bytes verbatim, so a digest match is exact;
    references then come from the source's resolved RPS, which stays correct
    even when the sub-stream itself is missing the parameter sets needed to
    parse slice headers.
    """
    by_digest = {_vcl_digest(source_vcl_bytes(layout, p.index)): p
                 for p in layout.pictures}
    for au in aus:
        pic = by_digest.get(_vcl_digest(au.vcl_nals)) if au.vcl_nals else None
        if pic is None:
            continue
        au.source_index = pic.index
        au.presentation = pic.display_index
        au.temporal_id = pic.temporal_id
        au.picture_type = pic.picture_type
        au.is_irap = pic.is_irap
        au.refs = tuple(pic.ref_indices)


@dataclass
class AURow:
    """One line of the dependency view: an AU and what it still needs."""
    index: int
    source_index: int | None
    presentation: int | None
    temporal_id: int | None
    picture_type: str
    is_irap: bool
    parameter_sets: list[str]        # parameter-set NALs riding with this AU
    config_active: bool              # VPS+SPS+PPS all seen at or before this AU
    refs: list[int]                  # source decode indices referenced
    missing_refs: list[int]          # referenced but absent from the sub-stream

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "source_index": self.source_index,
            "presentation": self.presentation,
            "temporal_id": self.temporal_id,
            "picture_type": self.picture_type,
            "is_irap": self.is_irap,
            "parameter_sets": self.parameter_sets,
            "config_active": self.config_active,
            "refs": self.refs,
            "missing_refs": self.missing_refs,
        }


def dependency_view(aus: list[AccessUnit]) -> list[AURow]:
    present = {au.source_index for au in aus if au.source_index is not None}
    rows: list[AURow] = []
    seen: set[int] = set()
    for au in aus:
        types = [_nal_type(n) for n in au.prefix_nals]
        seen.update(t for t in types if t in PARAM_TYPES)
        rows.append(AURow(
            index=au.index,
            source_index=au.source_index,
            presentation=au.presentation,
            temporal_id=au.temporal_id,
            picture_type=au.picture_type,
            is_irap=au.is_irap,
            parameter_sets=[PARAM_NAMES[t] for t in PARAM_TYPES if t in types],
            config_active=all(t in seen for t in PARAM_TYPES),
            refs=list(au.refs),
            missing_refs=[r for r in au.refs if r not in present],
        ))
    return rows


@dataclass
class FrameOut:
    presentation: int | None
    digest: str
    corrupt: bool
    key_frame: bool


@dataclass
class Probe:
    """Result of feeding a sub-stream to a fresh strict decoder, AU by AU."""
    ok: bool
    fault_au: int | None              # AU index at which the decoder refused
    fault_source_index: int | None
    error: dict | None
    frames: list[FrameOut] = field(default_factory=list)

    @property
    def emitted(self) -> list[int]:
        return [f.presentation for f in self.frames if f.presentation is not None]

    @property
    def frozen_at(self) -> int | None:
        """Last picture the decoder ever showed: where playback freezes."""
        em = self.emitted
        return em[-1] if em else None


def _take(frame) -> FrameOut:
    yuv = frame.to_ndarray(format="yuv444p")
    digest = hashlib.sha1(np.ascontiguousarray(yuv).tobytes()).hexdigest()
    pts = frame.pts
    presentation = pts if (pts is not None and pts < _UNALIGNED_PTS) else None
    return FrameOut(presentation=presentation, digest=digest,
                    corrupt=bool(frame.is_corrupt), key_frame=bool(frame.key_frame))


def probe(aus: list[AccessUnit], *, strict: bool = True) -> Probe:
    """Decode a list of access units in a fresh, strict decoder session."""
    ctx = av.CodecContext.create("hevc", "r")
    if strict:
        ctx.options = dict(STRICT_DECODER_OPTIONS)
    ctx.open()
    frames: list[FrameOut] = []
    error: dict | None = None
    fault_au: int | None = None
    for au in aus:
        if not au.vcl_nals and not au.prefix_nals:
            continue
        packet = av.Packet(au.to_bytes())
        pts = (au.presentation if au.presentation is not None
               else _UNALIGNED_PTS + au.index)
        packet.pts = pts
        packet.dts = au.index
        try:
            for frame in ctx.decode(packet):
                frames.append(_take(frame))
        except av.FFmpegError as exc:
            fault_au = au.index
            error = {"au_index": au.index, "source_index": au.source_index,
                     "error": type(exc).__name__, "message": str(exc)}
            break
    if error is None:
        try:
            for frame in ctx.decode(None):
                frames.append(_take(frame))
        except av.FFmpegError as exc:
            error = {"stage": "flush", "error": type(exc).__name__,
                     "message": str(exc)}
    fault_source = None
    if fault_au is not None:
        fault_source = next((a.source_index for a in aus if a.index == fault_au),
                            None)
    return Probe(ok=error is None, fault_au=fault_au,
                 fault_source_index=fault_source, error=error, frames=frames)


@dataclass
class Judgement:
    """Did an independent decode recover, and where is the first clean frame?"""
    recovered: bool
    first_clean: int | None
    probe: Probe
    mismatches: list[int] = field(default_factory=list)


def judge(aus: list[AccessUnit],
          reference: dict[int, DecodedPicture]) -> Judgement:
    """Classify an independent decode against the clean source decode.

    A frame only counts as clean when it is not corrupt, is not a repeat of
    the previously presented frame (decoder concealment), and is bit-exact
    with the reference at the same presentation time.
    """
    pr = probe(aus)
    if not pr.ok:
        return Judgement(False, None, pr)
    mismatches: list[int] = []
    first_clean: int | None = None
    prev_digest: str | None = None
    for f in sorted((f for f in pr.frames if f.presentation is not None),
                    key=lambda f: f.presentation):
        ref = reference.get(f.presentation)
        clean = (not f.corrupt and ref is not None and f.digest == ref.digest
                 and f.digest != prev_digest)
        prev_digest = f.digest
        if clean:
            if first_clean is None:
                first_clean = f.presentation
        else:
            mismatches.append(f.presentation)
    return Judgement(recovered=not mismatches and first_clean is not None,
                     first_clean=first_clean, probe=pr, mismatches=mismatches)


def _clone(au: AccessUnit, *, prefix: list[bytes] | None = None,
           vcl: list[bytes] | None = None) -> AccessUnit:
    return AccessUnit(
        index=au.index,
        prefix_nals=au.prefix_nals if prefix is None else prefix,
        vcl_nals=au.vcl_nals if vcl is None else vcl,
        source_index=au.source_index, presentation=au.presentation,
        temporal_id=au.temporal_id, picture_type=au.picture_type,
        is_irap=au.is_irap, refs=au.refs)


def _reindexed(aus: list[AccessUnit]) -> list[AccessUnit]:
    return [AccessUnit(i, au.prefix_nals, au.vcl_nals, au.source_index,
                       au.presentation, au.temporal_id, au.picture_type,
                       au.is_irap, au.refs)
            for i, au in enumerate(aus)]


def render(aus: list[AccessUnit]) -> bytes:
    return b"".join(au.to_bytes() for au in aus)


def _with_source_parameters(aus: list[AccessUnit],
                            layout: StreamLayout) -> list[AccessUnit]:
    """Same stream, but the first AU carries the source's active VPS/SPS/PPS.

    Nothing else is touched: this is the "只替换参数集" experiment.
    """
    first = next((au for au in aus if au.vcl_nals), None)
    if first is None or first.source_index is None:
        raise LocatorError("cannot locate the first picture in the source stream")
    correct = _parameter_sets(layout, first.source_index)
    prefix = [n.data for n in correct]
    prefix += [n for n in first.prefix_nals if _nal_type(n) not in PARAM_TYPES]
    return _reindexed([_clone(first, prefix=prefix) if au is first else au
                       for au in aus])


def _au_from_source(layout: StreamLayout, source_index: int) -> AccessUnit:
    pic = layout.pictures[source_index]
    au = AccessUnit(-1, [], source_vcl_bytes(layout, source_index))
    au.source_index = pic.index
    au.presentation = pic.display_index
    au.temporal_id = pic.temporal_id
    au.picture_type = pic.picture_type
    au.is_irap = pic.is_irap
    au.refs = tuple(pic.ref_indices)
    return au


def _insert_source_aus(aus: list[AccessUnit], layout: StreamLayout,
                       source_indices: list[int]) -> list[AccessUnit]:
    """Splice source pictures (VCL only, no parameter sets) into decode order."""
    wanted = sorted(source_indices)
    out: list[AccessUnit] = []
    for au in aus:
        while wanted and (au.source_index is None or wanted[0] < au.source_index):
            out.append(_au_from_source(layout, wanted.pop(0)))
        out.append(au)
    out.extend(_au_from_source(layout, i) for i in wanted)
    return _reindexed(out)


@dataclass
class WhatIf:
    """One single-change experiment on the dependency view."""
    action: str                       # "replace_parameter_sets" | "restore_reference"
    detail: dict
    applicable: bool
    recovered: bool
    first_clean: int | None
    errors: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"action": self.action, "detail": self.detail,
                "applicable": self.applicable, "recovered": self.recovered,
                "first_clean": self.first_clean, "errors": self.errors}


def what_if_replace_parameter_sets(aus: list[AccessUnit], layout: StreamLayout,
                                   reference: dict[int, DecodedPicture]) -> WhatIf:
    """Change *only* the parameter sets: give the first AU the source's active
    VPS/SPS/PPS.  Recovers exactly when the chain was broken by lost config."""
    try:
        candidate = _with_source_parameters(aus, layout)
    except LocatorError as exc:
        return WhatIf("replace_parameter_sets", {"reason": str(exc)},
                      applicable=False, recovered=False, first_clean=None)
    j = judge(candidate, reference)
    first = next(au for au in candidate if au.vcl_nals)
    return WhatIf(
        "replace_parameter_sets",
        {"injected": [PARAM_NAMES[t] for t in PARAM_TYPES],
         "at_source_index": first.source_index},
        applicable=True, recovered=j.recovered, first_clean=j.first_clean,
        errors=[j.probe.error] if j.probe.error else [])


def what_if_restore_reference(aus: list[AccessUnit], layout: StreamLayout,
                              reference: dict[int, DecodedPicture],
                              ref_source_index: int | None = None) -> WhatIf:
    """Change *only* one reference: carry the single source picture the first
    broken AU is missing.  Recovers exactly when a cropped enhancement frame
    was the root cause."""
    present = {au.source_index for au in aus if au.source_index is not None}
    target: AccessUnit | None = None
    missing: list[int] = []
    for au in aus:
        miss = [r for r in au.refs if r not in present]
        if miss:
            target, missing = au, miss
            break
    if target is None:
        return WhatIf("restore_reference",
                      {"reason": "no missing reference in the dependency view"},
                      applicable=False, recovered=False, first_clean=None)
    ref = ref_source_index if ref_source_index is not None else missing[0]
    candidate = _insert_source_aus(aus, layout, [ref])
    j = judge(candidate, reference)
    return WhatIf(
        "restore_reference",
        {"restored_source_index": ref,
         "restored_presentation": layout.pictures[ref].display_index,
         "needed_by_source_index": target.source_index},
        applicable=True, recovered=j.recovered, first_clean=j.first_clean,
        errors=[j.probe.error] if j.probe.error else [])


def _signature(aus: list[AccessUnit], pr: Probe) -> tuple | None:
    """Fault identity a minimised stream must preserve: which AU fails (by
    source picture), the decoder's error class, whether configuration was
    active at that AU, and whether it still references absent pictures."""
    if pr.ok:
        return None
    if pr.fault_au is None:
        return ("__flush__", pr.error["error"] if pr.error else None)
    au = next((a for a in aus if a.index == pr.fault_au), None)
    if au is None:
        return None
    present = {a.source_index for a in aus if a.source_index is not None}
    seen: set[int] = set()
    for a in aus:
        for n in a.prefix_nals:
            t = _nal_type(n)
            if t in PARAM_TYPES:
                seen.add(t)
        if a is au:
            break
    return (au.source_index, pr.error["error"] if pr.error else None,
            all(t in seen for t in PARAM_TYPES),
            any(r not in present for r in au.refs))


@dataclass
class Minimization:
    """The shortest sub-stream that still exhibits the same fault."""
    aus: list[AccessUnit]
    data: bytes
    au_count: int
    source_indices: list[int]
    signature: tuple
    probe: Probe

    def as_dict(self) -> dict:
        return {"au_count": self.au_count, "source_indices": self.source_indices,
                "bytes": len(self.data), "signature": list(self.signature),
                "still_fails": not self.probe.ok}


def minimize_failure(aus: list[AccessUnit]) -> Minimization:
    """Shrink a broken sub-stream while preserving its fault signature.

    Elements are whole AUs and per-AU prefix segments (the parameter sets and
    AUD riding with a picture).  Everything past the fault AU is dropped at
    once, then a greedy elimination removes any element whose absence keeps
    the same AU failing in the same way.
    """
    base = probe(aus)
    if base.ok:
        raise LocatorError("stream decodes; nothing to minimise")
    signature = _signature(aus, base)

    items: list[tuple[str, int]] = []
    for au in aus:
        if au.prefix_nals:
            items.append(("prefix", au.index))
        if au.vcl_nals:
            items.append(("picture", au.index))

    def rebuild(kept: list[tuple[str, int]]) -> list[AccessUnit]:
        have: dict[int, set[str]] = {}
        for kind, idx in kept:
            have.setdefault(idx, set()).add(kind)
        out = []
        for au in aus:
            kinds = have.get(au.index, set())
            if "picture" in kinds:
                out.append(_clone(au, prefix=au.prefix_nals
                                  if "prefix" in kinds else []))
            elif "prefix" in kinds:
                out.append(_clone(au, vcl=[]))
        return out

    def still_fails(kept: list[tuple[str, int]]) -> bool:
        candidate = rebuild(kept)
        if not any(au.vcl_nals for au in candidate):
            return False
        return _signature(candidate, probe(candidate)) == signature

    # Everything past the fault AU cannot influence the decoder before it.
    if base.fault_au is not None:
        kept = [it for it in items if it[1] <= base.fault_au]
        if not still_fails(kept):
            kept = list(items)
    else:
        kept = list(items)
    changed = True
    while changed:
        changed = False
        for it in list(kept):
            trial = [x for x in kept if x != it]
            if still_fails(trial):
                kept = trial
                changed = True
    final = rebuild(kept)
    final_probe = probe(final)
    pictures = [au for au in final if au.vcl_nals]
    return Minimization(
        aus=final, data=render(final), au_count=len(pictures),
        source_indices=[au.source_index for au in pictures],
        signature=signature, probe=final_probe)


@dataclass
class SegmentRequirement:
    """Whether one segment of a *legal* stream is required for full recovery."""
    label: str
    kind: str                       # "prefix" | "picture"
    au_index: int
    source_index: int | None
    required: bool

    def as_dict(self) -> dict:
        return {"label": self.label, "kind": self.kind, "au_index": self.au_index,
                "source_index": self.source_index, "required": self.required}


def required_segments(aus: list[AccessUnit],
                      reference: dict[int, DecodedPicture]) -> list[SegmentRequirement]:
    """Necessity check for a legal excerpt: a segment is required iff removing
    it alone breaks full recovery.  Redundant parameter sets (repeated config
    the decoder already holds) come out as *not* required."""
    if not judge(aus, reference).recovered:
        raise LocatorError(
            "stream does not fully recover; requirement analysis needs a legal stream")
    items: list[tuple[str, AccessUnit]] = []
    for au in aus:
        if au.prefix_nals:
            items.append(("prefix", au))
        if au.vcl_nals:
            items.append(("picture", au))

    def without(drop: tuple[str, AccessUnit]) -> list[AccessUnit]:
        out = []
        for au in aus:
            kinds = {k for k, a in items if a is au and (k, a) != drop}
            if "picture" in kinds:
                out.append(_clone(au, prefix=au.prefix_nals
                                  if "prefix" in kinds else []))
            elif "prefix" in kinds:
                out.append(_clone(au, vcl=[]))
        return out

    report: list[SegmentRequirement] = []
    for kind, au in items:
        if kind == "prefix":
            names = [NAL_NAMES.get(_nal_type(n), str(_nal_type(n)))
                     for n in au.prefix_nals]
            label = f"AU{au.index} prefix ({','.join(names)})"
        else:
            label = (f"AU{au.index} picture src {au.source_index} "
                     f"disp {au.presentation} tid {au.temporal_id}")
        recovered = judge(without((kind, au)), reference).recovered
        report.append(SegmentRequirement(
            label=label, kind=kind, au_index=au.index,
            source_index=au.source_index, required=not recovered))
    return report


@dataclass
class RecoveryPoint:
    """A picture before the fault that a player can independently jump to."""
    au_index: int
    source_index: int
    presentation: int
    needs_parameter_sets: bool
    verified: bool

    def as_dict(self) -> dict:
        return {"au_index": self.au_index, "source_index": self.source_index,
                "presentation": self.presentation,
                "needs_parameter_sets": self.needs_parameter_sets,
                "verified": self.verified}


def nearest_recovery_point(aus: list[AccessUnit], fault_au: int,
                           layout: StreamLayout,
                           reference: dict[int, DecodedPicture]) -> RecoveryPoint | None:
    """Nearest independently enterable picture *before* the fault AU.

    A candidate must be an IRAP whose first picture actually decodes on its
    own (verified against the reference); if the sub-stream carries no
    parameter sets at or before it, the source's sets are lent for the check
    and the point is flagged as needing them.
    """
    for i in range(min(fault_au, len(aus)) - 1, -1, -1):
        au = aus[i]
        if not au.vcl_nals or not au.is_irap or au.source_index is None:
            continue
        seen: set[int] = set()
        for prior in aus[:i + 1]:
            seen.update(t for t in PARAM_TYPES
                        if any(_nal_type(n) == t for n in prior.prefix_nals))
        has_params = all(t in seen for t in PARAM_TYPES)
        prefix = list(au.prefix_nals)
        if not has_params:
            prefix = ([n.data for n in _parameter_sets(layout, au.source_index)]
                      + [n for n in prefix if _nal_type(n) not in PARAM_TYPES])
        j = judge([_clone(au, prefix=prefix)], reference)
        if j.recovered:
            return RecoveryPoint(au_index=au.index, source_index=au.source_index,
                                 presentation=au.presentation,
                                 needs_parameter_sets=not has_params,
                                 verified=True)
    return None


@dataclass
class Repair:
    """The minimal fix, applied: never the whole source stream pasted back."""
    kind: str                       # "extractor" | "capability"
    data: bytes
    au_count: int
    added_source_indices: list[int]
    recovered: bool
    first_recovery: int | None

    def as_dict(self) -> dict:
        return {"kind": self.kind, "bytes": len(self.data),
                "au_count": self.au_count,
                "added_source_indices": self.added_source_indices,
                "recovered": self.recovered,
                "first_recovery": self.first_recovery}


def repair_stream(aus: list[AccessUnit], layout: StreamLayout,
                  reference: dict[int, DecodedPicture],
                  cause: Cause) -> Repair | None:
    """Apply the one fix the diagnosis proved out.

    Lost configuration is an *extractor* bug: re-emit with the active
    VPS/SPS/PPS prepended (no AU added).  A cropped reference is a
    *capability* misjudgement: the cut was not independently decodable, so
    carry exactly the missing reference closure (the preroll the request
    needed).  Either way the result stays a strict subset of the source.
    """
    if cause == Cause.MISSING_PARAMETER_SETS:
        fixed = _with_source_parameters(aus, layout)
        kind, added = "extractor", []
    elif cause == Cause.MISSING_REFERENCE:
        present = {au.source_index for au in aus if au.source_index is not None}
        want: set[int] = set()
        frontier = [r for au in aus for r in au.refs
                    if r not in present and r not in want]
        while frontier:
            ref = frontier.pop()
            if ref in present or ref in want:
                continue
            want.add(ref)
            frontier.extend(x for x in layout.pictures[ref].ref_indices
                            if x not in present and x not in want)
        fixed = _insert_source_aus(aus, layout, sorted(want))
        kind, added = "capability", sorted(want)
    else:
        return None
    j = judge(fixed, reference)
    return Repair(kind=kind, data=render(fixed),
                  au_count=sum(1 for au in fixed if au.vcl_nals),
                  added_source_indices=added,
                  recovered=j.recovered, first_recovery=j.first_clean)


@dataclass
class Diagnosis:
    probe: Probe
    cause: Cause
    view: list[AURow]
    experiments: list[WhatIf]
    minimal: Minimization | None
    recovery_point: RecoveryPoint | None
    repair: Repair | None

    def as_dict(self) -> dict:
        return {
            "cause": self.cause.value,
            "probe": {
                "ok": self.probe.ok,
                "fault_au": self.probe.fault_au,
                "fault_source_index": self.probe.fault_source_index,
                "error": self.probe.error,
                "emitted": self.probe.emitted,
                "frozen_at": self.probe.frozen_at,
            },
            "view": [row.as_dict() for row in self.view],
            "experiments": [e.as_dict() for e in self.experiments],
            "minimal": self.minimal.as_dict() if self.minimal else None,
            "recovery_point": (self.recovery_point.as_dict()
                               if self.recovery_point else None),
            "repair": self.repair.as_dict() if self.repair else None,
        }


def diagnose(sub_blob: bytes, source_blob: bytes, *,
             layout: StreamLayout | None = None,
             reference: dict[int, DecodedPicture] | None = None) -> Diagnosis:
    """Locate the broken link of a sub-stream, access unit by access unit."""
    layout = layout or parse_stream(source_blob)
    if reference is None:
        reference = decode_reference(source_blob, layout)
    aus = split_access_units(sub_blob)
    align_to_source(aus, layout)
    view = dependency_view(aus)
    pr = probe(aus)
    if pr.ok:
        return Diagnosis(pr, Cause.NONE, view, [], None, None, None)

    experiments = [
        what_if_replace_parameter_sets(aus, layout, reference),
        what_if_restore_reference(aus, layout, reference),
    ]
    by_action = {e.action: e for e in experiments}
    params_fix = by_action["replace_parameter_sets"].recovered
    ref_fix = by_action["restore_reference"].recovered
    if params_fix and not ref_fix:
        cause = Cause.MISSING_PARAMETER_SETS
    elif ref_fix and not params_fix:
        cause = Cause.MISSING_REFERENCE
    else:
        cause = Cause.UNKNOWN

    minimal = minimize_failure(aus)
    fault_au = pr.fault_au if pr.fault_au is not None else len(aus)
    recovery = nearest_recovery_point(aus, fault_au, layout, reference)
    repair = repair_stream(aus, layout, reference, cause)
    return Diagnosis(pr, cause, view, experiments, minimal, recovery, repair)
