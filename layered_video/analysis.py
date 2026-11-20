"""One request → plan → sub-stream → clean-decode verification.

Shared by the CLI and the preview server so both report the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bitstream import StreamLayout, parse_stream
from .decode import (
    DecodedPicture, SwitchEvaluation, decode_reference, decode_substream,
    evaluate_stream,
)
from .extractor import (
    ExtractError, ExtractedStream, ExtractionPlan, Policy, build_substream,
    plan_extraction,
)

# How each policy is presented.  "forced" deliberately emits a stream the
# extractor would reject, so the verifier can show that the decoder's own
# concealment does not count as a recovery.
POLICY_LABELS = {
    "defer": "wait for the next switchable position",
    "preroll": "pre-reserve the needed references",
    "naive": "naive cut (rejected)",
    "forced": "forced naive cut (unsafe, for comparison)",
}


@dataclass
class Analysis:
    requested_presentation: int
    plan: ExtractionPlan
    layer_set: list[int]
    policy: str
    stream: ExtractedStream | None = None
    evaluation: SwitchEvaluation | None = None
    error: str | None = None
    source_psnr: dict[int, float] = field(default_factory=dict)
    _layout: StreamLayout | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.evaluation is not None and self.error is None

    def summary(self) -> dict:
        layout = self._layout
        preroll_display = ([layout.pictures[i].display_index
                            for i in self.plan.preroll_decode_indices]
                           if layout is not None else self.plan.preroll_decode_indices)
        base = {
            "policy": self.policy,
            "policy_label": POLICY_LABELS.get(self.policy, self.policy),
            "requested_presentation": self.requested_presentation,
            "layer_set": self.layer_set,
            "legal": self.plan.is_legal,
            "switch_presentation": self.plan.switch_presentation,
            "deferred_by": self.plan.deferred_by,
            "preroll": preroll_display,
            "preroll_decode_indices": self.plan.preroll_decode_indices,
            "kept_count": self.plan.kept_count,
            "missing": [m.as_dict() for m in self.plan.missing],
            "error": self.error,
        }
        if self.stream is not None:
            base["bytes_total"] = self.stream.total_bytes
            base["bytes_pictures"] = self.stream.picture_bytes
            base["bytes_parameters"] = self.stream.parameter_bytes
            base["picture_count"] = self.stream.access_unit_count
        if self.evaluation is not None:
            ev = self.evaluation
            enhancement = set(self.layer_set) - {0}
            base.update({
                "first_recovery": ev.first_recovery,
                "decode_errors": ev.decode_errors,
                # Pre-roll carries real enhancement pictures that display before
                # the requested switch; say so rather than hiding the cost.
                "enhancement_before_switch": list(ev.preroll),
                "counts": {
                    "recovered": sum(1 for f in ev.frames if f.recovers),
                    "duplicate": sum(1 for f in ev.frames if f.duplicate),
                    "corrupt": sum(1 for f in ev.frames if f.corrupt),
                    "missing": sum(1 for f in ev.frames if not f.decoded),
                    "base": sum(1 for f in ev.frames
                                if f.decoded and f.temporal_id not in enhancement),
                },
            })
            if self.source_psnr:
                base["source_psnr"] = {str(k): round(v, 2)
                                       for k, v in self.source_psnr.items()}
        return base

    def frames(self) -> list[dict]:
        return [f.as_dict() for f in (self.evaluation.frames if self.evaluation else [])]


def timeline_for(layout: StreamLayout, switch: int, layer_set: list[int],
                 low_layers: list[int],
                 extra_presentations: list[int] | None = None) -> list[int]:
    """Presentation slots the sub-stream is supposed to deliver.

    Before the switch the base layers are played; from the switch on, the
    requested layers are.  Anything else was deliberately left in the low
    bit-rate segment and is not counted as missing.
    """
    lows = set(low_layers)
    highs = set(layer_set)
    extra = set(extra_presentations or [])
    return sorted(
        p.display_index for p in layout.pictures
        if ((p.display_index < switch and p.temporal_id in lows)
            or (p.display_index >= switch and p.temporal_id in highs)
            or p.display_index in extra)
    )


def analyse(blob: bytes, layout: StreamLayout, presentation: int,
            layer_set: list[int], policy: str = "preroll",
            low_layer_set: list[int] | None = None,
            reference: dict[int, DecodedPicture] | None = None,
            verify: bool = True) -> Analysis:
    low = list(low_layer_set if low_layer_set is not None else [0])
    forced = policy == "forced"
    effective = Policy.NAIVE if forced else Policy(policy)
    try:
        plan = plan_extraction(layout, presentation, layer_set, effective,
                               low_layer_set=low)
    except ExtractError as exc:
        plan = ExtractionPlan(
            policy=effective, layer_set=sorted(set(layer_set)),
            low_layers=sorted(low), requested_presentation=presentation,
            switch_presentation=presentation, switch_decode_index=-1,
            kept_decode_indices=[], preroll_decode_indices=[], deferred_by=0,
            missing=[], total_pictures=layout.frame_count,
        )
        return Analysis(presentation, plan, sorted(set(layer_set)), policy,
                        error=str(exc))

    if not forced and plan.missing:
        return Analysis(
            presentation, plan, sorted(set(layer_set)), policy,
            error=("refusing to emit: a kept picture references a frame that "
                   "was cut away"),
            _layout=layout,
        )

    if forced:
        saved = plan.missing
        plan.missing = []
        try:
            stream = build_substream(layout, blob, plan)
        finally:
            plan.missing = saved
    else:
        stream = build_substream(layout, blob, plan)

    result = Analysis(presentation, plan, sorted(set(layer_set)), policy,
                      stream=stream, _layout=layout)
    if not verify:
        return result

    if reference is None:
        reference = decode_reference(blob, layout)
    sub_layout = parse_stream(stream.data)
    decoded = decode_substream(stream.data, layout=sub_layout,
                               presentations=stream.presentations,
                               strict=not forced)
    preroll_presentations = [layout.pictures[i].display_index
                             for i in plan.preroll_decode_indices]
    timeline = timeline_for(layout, plan.switch_presentation, layer_set, low,
                            preroll_presentations)
    evaluation = evaluate_stream(
        layout, reference, decoded,
        policy=policy,
        requested_presentation=presentation,
        switch_presentation=plan.switch_presentation,
        layer_set=layer_set, low_layers=low, legal=plan.is_legal,
        deferred_by=plan.deferred_by,
        preroll=preroll_presentations,
        missing=[m.as_dict() for m in plan.missing],
        bytes_total=stream.total_bytes,
        bytes_pictures=stream.picture_bytes,
        picture_count=stream.access_unit_count,
        timeline=timeline,
        preroll_presentations=preroll_presentations,
    )
    result.evaluation = evaluation
    return result
