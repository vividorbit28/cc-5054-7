"""Independent decoding sessions and measurement.

Nothing here shares state with the extractor: a sub-stream is handed as raw
Annex-B packets to a *fresh* decoder, and its output is compared frame by frame
against a clean decode of the full stream (or against the source frames).

Three failure modes are explicitly classified, because the task is to prove
that a layer upgrade really recovered, not that the decoder kept producing
pictures:

* ``corrupt``     - the decoder flagged the picture (``Frame.is_corrupt``);
* ``duplicate``   - the picture is pixel-identical to the previously presented
  one, which is what concealment looks like when a reference is missing;
* ``wrong``       - the picture decoded cleanly but does not match the clean
  full-stream decode at the same presentation time.

A picture only counts as "recovered" when none of those hold and the measured
PSNR against the reference meets the target threshold.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import av
import numpy as np

from .bitstream import StreamLayout, parse_stream

# Strict decoding: turn bitstream/conformance violations into exceptions
# instead of letting libavcodec conceal them.
STRICT_DECODER_OPTIONS = {
    "err_detect": "crccheck+bitstream+buffer+explode",
    "ec": "0",            # error_concealment = 0
    "threads": "1",       # deterministic frame order
}


class DecoderError(RuntimeError):
    """Raised when a clean decoder session refuses a sub-stream."""


@dataclass
class DecodedPicture:
    presentation: int
    decode_index: int
    picture_type: str
    key_frame: bool
    corrupt: bool
    yuv: np.ndarray            # yuv444p planes, shape (3, H, W)
    digest: str

    def as_dict(self) -> dict:
        return {
            "presentation": self.presentation,
            "decode_index": self.decode_index,
            "picture_type": self.picture_type,
            "key_frame": self.key_frame,
            "corrupt": self.corrupt,
            "digest": self.digest,
        }


@dataclass
class DecodeResult:
    pictures: list[DecodedPicture]
    errors: list[dict] = field(default_factory=list)
    refused_at_decode_index: int | None = None

    @property
    def by_presentation(self) -> dict[int, DecodedPicture]:
        return {p.presentation: p for p in self.pictures}


def _access_unit_bytes(layout: StreamLayout, blob: bytes, picture) -> bytes:
    prefix = []
    k = picture.first_nal_index - 1
    while k >= 0 and not layout.nals[k].is_vcl:
        prefix.insert(0, layout.nals[k])
        k -= 1
    first = layout.nals[picture.first_nal_index]
    last = layout.nals[picture.last_nal_index]
    return (b"".join(b"\x00\x00\x01" + n.data for n in prefix)
            + blob[first.offset:last.offset + last.size])


def decode_substream(blob: bytes, *, layout: StreamLayout | None = None,
                     presentations: list[int] | None = None,
                     strict: bool = True) -> DecodeResult:
    """Decode an Annex-B sub-stream in its own decoder context.

    Each packet is one access unit, tagged with the *source* presentation index
    it will show at, so the returned pictures line up with the full-stream
    decode even though decode order differs from presentation order.  When
    pictures were dropped, the sub-stream's own presentation numbering is
    compressed, so ``presentations`` (from ``ExtractedStream``) must be passed.
    """
    layout = layout or parse_stream(blob)
    if presentations is not None and len(presentations) != layout.frame_count:
        raise DecoderError(
            f"got {len(presentations)} presentation tags for "
            f"{layout.frame_count} pictures")
    ctx = av.CodecContext.create("hevc", "r")
    if strict:
        ctx.options = dict(STRICT_DECODER_OPTIONS)
    ctx.open()

    pictures: list[DecodedPicture] = []
    errors: list[dict] = []
    refused_at: int | None = None

    def take(frame, expected_presentation: int):
        yuv = frame.to_ndarray(format="yuv444p")
        digest = hashlib.sha1(np.ascontiguousarray(yuv).tobytes()).hexdigest()
        pictures.append(DecodedPicture(
            presentation=expected_presentation,
            decode_index=len(pictures),
            picture_type=str(frame.pict_type),
            key_frame=bool(frame.key_frame),
            corrupt=bool(frame.is_corrupt),
            yuv=yuv,
            digest=digest,
        ))

    # Map each picture's decode-order position to the presentation tag to carry
    # through the decoder: FFmpeg returns a frame when it is output, but the PTS
    # it hands back is the PTS of the *packet* that completed it.  With
    # reordering, output order equals presentation order, so tag packets by
    # decode index and look the presentation up from the output frame's PTS.
    tag_for_decode: dict[int, int] = {}
    if presentations is not None:
        for pos, picture in enumerate(layout.pictures):
            tag_for_decode[picture.index] = presentations[pos]

    pending_tags: dict[int, int] = {}

    def take_by_pts(frame):
        pts = frame.pts if frame.pts is not None else -1
        if pts in pending_tags:
            presentation = pending_tags[pts]
        elif presentations is None:
            presentation = pts
        else:
            presentation = -1
        take(frame, presentation)

    for pos, picture in enumerate(layout.pictures):
        payload = blob[layout.nals[picture.first_nal_index].offset:
                       layout.nals[picture.last_nal_index].offset
                       + layout.nals[picture.last_nal_index].size]
        packet = av.Packet(_access_unit_bytes(layout, blob, picture))
        packet.pts = picture.index
        packet.dts = picture.index
        pending_tags[picture.index] = (
            tag_for_decode.get(picture.index, picture.display_index))
        try:
            for frame in ctx.decode(packet):
                take_by_pts(frame)
        except av.FFmpegError as exc:
            errors.append({
                "decode_index": picture.index,
                "presentation": pending_tags[picture.index],
                "error": type(exc).__name__,
                "message": str(exc),
            })
            refused_at = picture.index
            break
    if refused_at is None:
        try:
            for frame in ctx.decode(None):
                take_by_pts(frame)
        except av.FFmpegError as exc:
            errors.append({"stage": "flush", "error": type(exc).__name__,
                           "message": str(exc)})
    return DecodeResult(pictures=pictures, errors=errors,
                        refused_at_decode_index=refused_at)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(((a - b) ** 2).mean())
    if mse == 0.0:
        return 99.0
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def decode_reference(blob: bytes, layout: StreamLayout | None = None) -> dict[int, DecodedPicture]:
    """Clean decode of the complete stream, keyed by presentation index."""
    result = decode_substream(blob, layout=layout, strict=True)
    if result.errors:
        raise DecoderError(f"reference decode failed: {result.errors[0]}")
    return result.by_presentation


@dataclass
class FrameVerdict:
    presentation: int
    temporal_id: int
    decoded: bool
    corrupt: bool
    duplicate: bool
    psnr: float
    recovers: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "presentation": self.presentation,
            "temporal_id": self.temporal_id,
            "decoded": self.decoded,
            "corrupt": self.corrupt,
            "duplicate": self.duplicate,
            "psnr": round(self.psnr, 3),
            "recovers": self.recovers,
            "reason": self.reason,
        }


@dataclass
class SwitchEvaluation:
    policy: str
    requested_presentation: int
    switch_presentation: int
    layer_set: list[int]
    legal: bool
    deferred_by: int
    preroll: list[int]
    missing: list[dict]
    bytes_total: int
    bytes_pictures: int
    picture_count: int
    first_recovery: int | None
    target_psnr: float
    frames: list[FrameVerdict]
    decode_errors: list[dict]

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "requested_presentation": self.requested_presentation,
            "switch_presentation": self.switch_presentation,
            "layer_set": self.layer_set,
            "legal": self.legal,
            "deferred_by": self.deferred_by,
            "preroll": self.preroll,
            "missing": self.missing,
            "bytes_total": self.bytes_total,
            "bytes_pictures": self.bytes_pictures,
            "picture_count": self.picture_count,
            "first_recovery": self.first_recovery,
            "target_psnr": self.target_psnr,
            "decode_errors": self.decode_errors,
            "frames": [f.as_dict() for f in self.frames],
        }


def evaluate_stream(layout: StreamLayout, reference: dict[int, DecodedPicture],
                    decoded: DecodeResult, *, policy: str,
                    requested_presentation: int, switch_presentation: int,
                    layer_set: list[int], low_layers: list[int] | None = None,
                    legal: bool, deferred_by: int,
                    preroll: list[int], missing: list[dict],
                    bytes_total: int, bytes_pictures: int, picture_count: int,
                    timeline: list[int] | None = None,
                    preroll_presentations: list[int] | None = None,
                    target_psnr: float = 38.0) -> SwitchEvaluation:
    """Classify every presentation slot of the sub-stream.

    A picture only counts as recovering the target quality if it is actually
    present, decoded cleanly, is *not* a repeat of the previous picture, and
    matches the clean full-stream decode.  Decoder concealment and frame
    duplication are therefore never mistaken for a successful upgrade.
    """
    low = set(low_layers or [0])
    enhancement = set(layer_set) - low
    decoded_by_pres = {p.presentation: p for p in decoded.pictures}
    slots = timeline if timeline is not None else sorted(decoded_by_pres)
    preroll_set = set(preroll_presentations or [])
    verdicts: list[FrameVerdict] = []
    previous_digest: str | None = None
    first_recovery: int | None = None

    for presentation in sorted(slots):
        tid = layout.temporal_id_of(presentation)
        pic = decoded_by_pres.get(presentation)
        if pic is None:
            verdicts.append(FrameVerdict(
                presentation, tid, False, False, False, 0.0, False,
                "not present in the sub-stream (layer dropped)"))
            previous_digest = None
            continue
        ref = reference.get(presentation)
        precarried = presentation in preroll_set
        duplicate = pic.digest == previous_digest
        previous_digest = pic.digest
        if ref is None:
            verdicts.append(FrameVerdict(presentation, tid, True, pic.corrupt,
                                         duplicate, 0.0, False,
                                         "no reference frame at this time"))
            continue
        score = psnr(pic.yuv, ref.yuv)
        if not legal:
            # The stream was emitted despite a broken reference chain.  Whatever
            # the decoder produced here (concealment, repeats, or even a frame
            # that happens to match) cannot be trusted as a recovery.
            reason = "illegal sub-stream: references were cut away"
        elif pic.corrupt:
            reason = "decoder flagged the picture as corrupt"
        elif duplicate:
            reason = "picture repeats the previous frame (concealment)"
        elif score < target_psnr:
            reason = f"below target quality ({score:.1f} dB < {target_psnr:.1f} dB)"
        elif tid not in enhancement:
            reason = "base layer (target enhancement layer not reached)"
        elif precarried:
            reason = "pre-roll reference (decoded early, before the switch)"
        else:
            reason = "clean enhancement decode matching the reference"
        recovers = (legal and not pic.corrupt and not duplicate
                    and score >= target_psnr and tid in enhancement
                    and not precarried)
        if recovers and first_recovery is None and presentation >= requested_presentation:
            first_recovery = presentation
        verdicts.append(FrameVerdict(presentation, tid, True, pic.corrupt,
                                     duplicate, score, recovers, reason))

    return SwitchEvaluation(
        policy=policy, requested_presentation=requested_presentation,
        switch_presentation=switch_presentation, layer_set=list(layer_set),
        legal=legal, deferred_by=deferred_by, preroll=list(preroll),
        missing=list(missing), bytes_total=bytes_total,
        bytes_pictures=bytes_pictures, picture_count=picture_count,
        first_recovery=first_recovery, target_psnr=target_psnr,
        frames=verdicts, decode_errors=list(decoded.errors),
    )
