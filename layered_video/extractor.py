"""Layer-aware sub-stream extraction.

Given a parsed HEVC elementary stream, build a *legal* sub-bitstream for a
caller-chosen layer upgrade, and report precisely which references are absent
when the request cannot be honoured.

A request is "at presentation time T, switch from the base layer to layers
0..L".  That means: every picture with presentation index < T that the base
layer would have shown, plus every picture with presentation index >= T whose
temporal id is <= L.  B-pyramid reordering means the switch is *not* a pure
presentation-order cut: high-layer pictures with presentation index >= T can
reference lower-layer pictures that sit *after* T in presentation order but
*before* T in decode order, and some of those lower-layer pictures are
themselves inter-coded and need their own references.  Those references are
exactly the "enhancement frames" that a naive cut drops.

Policies:

* ``defer``   - move the switch to the next presentation position whose
  sub-stream is self-contained with base layers only before it.  Playback
  reaches the switch no earlier than requested, never with broken references.
* ``preroll`` - keep the switch exactly where asked, and additionally carry the
  lowest decoding order closure of missing references.  Bytes are spent up
  front so the upgrade lands on time.

The naive policy is provided only to be *rejected*: ``build_substream`` refuses
to emit a stream whose references are not all present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .bitstream import (
    NAL_AUD, NAL_PPS, NAL_SPS, NAL_VPS, StreamLayout, parse_stream,
)


class Policy(str, Enum):
    NAIVE = "naive"        # cut at exactly T, keep only the layer set
    DEFER = "defer"        # slide the switch later until self-contained
    PREROLL = "preroll"    # keep the references the exact switch needs


class ExtractError(ValueError):
    """Raised when an extraction request cannot be satisfied safely."""


@dataclass(frozen=True)
class MissingDependency:
    picture_display: int
    picture_temporal_id: int
    missing_display: int
    missing_temporal_id: int
    missing_decode_index: int

    def as_dict(self) -> dict:
        return {
            "picture_display": self.picture_display,
            "picture_temporal_id": self.picture_temporal_id,
            "missing_display": self.missing_display,
            "missing_temporal_id": self.missing_temporal_id,
            "missing_decode_index": self.missing_decode_index,
        }


@dataclass
class ExtractionPlan:
    policy: Policy
    layer_set: list[int]
    low_layers: list[int]
    requested_presentation: int
    switch_presentation: int      # presentation index actually used
    switch_decode_index: int      # decode index of the switch picture (-1 if none)
    kept_decode_indices: list[int]
    preroll_decode_indices: list[int]
    deferred_by: int
    missing: list[MissingDependency]
    total_pictures: int

    @property
    def is_legal(self) -> bool:
        return not self.missing

    @property
    def kept_count(self) -> int:
        return len(self.kept_decode_indices)

    def as_dict(self) -> dict:
        return {
            "policy": self.policy.value,
            "layer_set": self.layer_set,
            "low_layers": self.low_layers,
            "requested_presentation": self.requested_presentation,
            "switch_presentation": self.switch_presentation,
            "switch_decode_index": self.switch_decode_index,
            "kept_decode_indices": self.kept_decode_indices,
            "preroll_decode_indices": self.preroll_decode_indices,
            "deferred_by": self.deferred_by,
            "kept_count": len(self.kept_decode_indices),
            "total_pictures": self.total_pictures,
            "legal": self.is_legal,
            "missing": [m.as_dict() for m in self.missing],
        }


@dataclass
class ExtractedStream:
    data: bytes
    plan: ExtractionPlan
    picture_bytes: int
    parameter_bytes: int
    access_unit_count: int
    per_picture_bytes: dict[int, int] = field(default_factory=dict)
    # Source presentation index for each emitted picture, in decode order.  The
    # sub-stream's own renumbered presentation order is *not* the source order
    # once pictures are dropped, so the decoder must be fed these explicit PTS.
    presentations: list[int] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return len(self.data)


def _prefix_nals(layout: StreamLayout, picture) -> list:
    out = []
    k = picture.first_nal_index - 1
    while k >= 0 and not layout.nals[k].is_vcl:
        out.insert(0, layout.nals[k])
        k -= 1
    return out


def _nal_bytes(nal) -> bytes:
    return b"\x00\x00\x01" + nal.data


def _picture_bytes(layout: StreamLayout, blob: bytes, picture) -> bytes:
    first = layout.nals[picture.first_nal_index]
    last = layout.nals[picture.last_nal_index]
    return blob[first.offset:last.offset + last.size]


def _parameter_sets(layout: StreamLayout, decode_index: int) -> list:
    """Most recent VPS/SPS/PPS at or before a picture, for sub-stream headers."""
    want = {NAL_VPS: None, NAL_SPS: None, NAL_PPS: None}
    for p in layout.pictures[:decode_index]:
        for nal in _prefix_nals(layout, p):
            if nal.nal_type in want:
                want[nal.nal_type] = nal
    if 0 <= decode_index < len(layout.pictures):
        for nal in _prefix_nals(layout, layout.pictures[decode_index]):
            if nal.nal_type in want:
                want[nal.nal_type] = nal
    return [want[t] for t in (NAL_VPS, NAL_SPS, NAL_PPS) if want[t] is not None]


def _missing_for(layout: StreamLayout, kept: set[int]) -> list[MissingDependency]:
    missing = []
    for idx in sorted(kept):
        for ref in layout.pictures[idx].ref_indices:
            if ref not in kept:
                missing.append(MissingDependency(
                    picture_display=layout.pictures[idx].display_index,
                    picture_temporal_id=layout.pictures[idx].temporal_id,
                    missing_display=layout.pictures[ref].display_index,
                    missing_temporal_id=layout.pictures[ref].temporal_id,
                    missing_decode_index=ref,
                ))
    return missing


def _select(layout: StreamLayout, switch: int, layer_set: set[int],
            low_layers: set[int], extra: set[int] | None = None) -> set[int]:
    """Pictures a presentation switch at ``switch`` needs."""
    extra = extra or set()
    kept: set[int] = set()
    for p in layout.pictures:
        if p.display_index < switch:
            if p.temporal_id in low_layers or p.index in extra:
                kept.add(p.index)
        elif p.temporal_id in layer_set:
            kept.add(p.index)
    return kept


def _preroll_closure(layout: StreamLayout, kept: set[int], switch: int,
                     layer_set: set[int], low_layers: set[int]) -> set[int]:
    """Lowest closure of references a presentation switch drops.

    Only references that are themselves enhancement (temporal id outside the
    low set) become preroll pictures; base-layer references are included by the
    normal selection.  Following the closure matters because an enhancement
    anchor can itself depend on an earlier enhancement picture.
    """
    extra: set[int] = set()
    stack = []
    for idx in sorted(kept):
        pic = layout.pictures[idx]
        for ref in pic.ref_indices:
            r = layout.pictures[ref]
            if ref not in kept and r.temporal_id not in low_layers:
                stack.append(ref)
    while stack:
        idx = stack.pop()
        if idx in extra or idx in kept:
            continue
        extra.add(idx)
        for ref in layout.pictures[idx].ref_indices:
            r = layout.pictures[ref]
            if r.display_index < switch and r.temporal_id not in low_layers \
                    and ref not in kept and ref not in extra:
                stack.append(ref)
    return extra


def plan_extraction(layout: StreamLayout, presentation: int, layer_set: list[int],
                    policy: Policy = Policy.PREROLL,
                    low_layer_set: list[int] | None = None) -> ExtractionPlan:
    n = layout.frame_count
    if not 0 <= presentation <= n:
        raise ExtractError(f"presentation index {presentation} outside 0..{n}")
    target = set(layer_set)
    if not target:
        raise ExtractError("layer set is empty")
    unknown = target - set(layout.temporal_ids)
    if unknown:
        raise ExtractError(
            f"layer(s) {sorted(unknown)} not present; stream has {layout.temporal_ids}")
    low = set(low_layer_set) if low_layer_set is not None else {0}
    if not low <= set(layout.temporal_ids):
        raise ExtractError(f"low layer set {sorted(low)} not present in stream")
    if not low <= target:
        raise ExtractError("low layers must be a subset of the requested layers")

    by_display = sorted(layout.pictures, key=lambda p: p.display_index)

    def plan_at(switch: int, kept: set[int], extra: set[int]) -> ExtractionPlan:
        # The upgrade is only *visible* at the first kept picture at or after the
        # requested time whose temporal id is an enhancement layer.  Deferring can
        # land on a gap (a tid-3 picture the layer set excludes) and the base
        # layer keeps being shown, so reporting the search cursor would claim a
        # quality recovery the viewer never sees.
        enhancement = target - low
        shown = sorted(
            (layout.pictures[i] for i in kept
             if layout.pictures[i].display_index >= presentation
             and layout.pictures[i].temporal_id in enhancement),
            key=lambda p: p.display_index)
        real_switch = shown[0] if shown else None
        return ExtractionPlan(
            policy=policy, layer_set=sorted(target), low_layers=sorted(low),
            requested_presentation=presentation,
            switch_presentation=(real_switch.display_index if real_switch else switch),
            switch_decode_index=real_switch.index if real_switch else -1,
            kept_decode_indices=sorted(kept),
            preroll_decode_indices=sorted(extra),
            deferred_by=(real_switch.display_index - presentation) if real_switch
                        else (switch - presentation),
            missing=_missing_for(layout, kept),
            total_pictures=n,
        )

    if policy == Policy.NAIVE:
        kept = _select(layout, presentation, target, low)
        return plan_at(presentation, kept, set())

    if policy == Policy.PREROLL:
        kept = _select(layout, presentation, target, low)
        extra = _preroll_closure(layout, kept, presentation, target, low)
        kept |= extra
        return plan_at(presentation, kept, extra)

    # DEFER: find the first presentation index at or after the request whose
    # base-layer prefix plus target layers is self-contained.
    switch = presentation
    while switch <= n:
        kept = _select(layout, switch, target, low)
        if not _missing_for(layout, kept):
            return plan_at(switch, kept, set())
        switch += 1
    raise ExtractError("no later presentation position is self-contained")


def build_substream(layout: StreamLayout, blob: bytes, plan: ExtractionPlan) -> ExtractedStream:
    """Serialise a plan to Annex-B, copying source bytes verbatim."""
    if plan.missing:
        first = plan.missing[0]
        raise ExtractError(
            "refusing to emit a stream with missing references: "
            f"{len(plan.missing)} picture(s) reference an absent frame, first is "
            f"display {first.picture_display} (tid {first.picture_temporal_id}) "
            f"needing display {first.missing_display} (tid {first.missing_temporal_id})")
    kept = plan.kept_decode_indices
    if not kept:
        raise ExtractError("nothing to extract")

    out = bytearray()
    parameter_bytes = 0
    for nal in _parameter_sets(layout, kept[0]):
        raw = _nal_bytes(nal)
        out += raw
        parameter_bytes += len(raw)

    per_picture: dict[int, int] = {}
    presentations: list[int] = []
    for idx in kept:
        pic = layout.pictures[idx]
        # Repeat the source's AUD / parameter-set prefixes so the sub-stream is
        # self-delimiting wherever it is injected.
        for nal in _prefix_nals(layout, pic):
            if nal.nal_type in (NAL_AUD, NAL_VPS, NAL_SPS, NAL_PPS):
                raw = _nal_bytes(nal)
                out += raw
                parameter_bytes += len(raw)
        raw = _picture_bytes(layout, blob, pic)
        per_picture[idx] = len(raw)
        presentations.append(pic.display_index)
        out += raw

    return ExtractedStream(
        data=bytes(out), plan=plan,
        picture_bytes=sum(per_picture.values()),
        parameter_bytes=parameter_bytes,
        access_unit_count=len(kept),
        per_picture_bytes=per_picture,
        presentations=presentations,
    )


def extract(blob: bytes, presentation: int, layer_set: list[int],
            policy: Policy = Policy.PREROLL,
            low_layer_set: list[int] | None = None) -> tuple[StreamLayout, ExtractedStream]:
    layout = parse_stream(blob)
    plan = plan_extraction(layout, presentation, layer_set, policy, low_layer_set)
    return layout, build_substream(layout, blob, plan)
