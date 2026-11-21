"""Command-line extraction and verification.

    python -m layered_video.cli info  data/sample_4layer.h265
    python -m layered_video.cli plan  data/sample_4layer.h265 --at 6 --layers 0,1,2
    python -m layered_video.cli verify data/sample_4layer.h265 --at 6 --layers 0,1,2
    python -m layered_video.cli extract data/sample_4layer.h265 --at 6 --layers 0,1,2 --out sub.h265
    python -m layered_video.cli diagnose sub.h265 --source data/sample_4layer.h265
"""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import POLICY_LABELS, analyse
from .bitstream import parse_stream
from .extractor import Policy
from .locator import diagnose


def _layers(text: str) -> list[int]:
    return sorted({int(x) for x in text.replace(" ", "").split(",") if x != ""})


def _print_layout(layout) -> None:
    print(f"pictures       {layout.frame_count}  ({layout.gop_count} GOPs)")
    print(f"temporal ids   {layout.temporal_ids}")
    counts = {t: sum(1 for p in layout.pictures if p.temporal_id == t)
              for t in layout.temporal_ids}
    print(f"per layer      {counts}")
    print(f"IRAP (decode)  {layout.irap_positions()}")
    print()
    print(" dec  disp  tid  type  gop  refs(decode)")
    for p in layout.pictures[:24]:
        print(f" {p.index:3d}  {p.display_index:4d}  {p.temporal_id:3d}  "
              f"{p.picture_type:4s}  {p.gop:3d}  {p.ref_indices}")
    if layout.frame_count > 24:
        print(f" ... {layout.frame_count - 24} more")


def _print_diagnosis(d) -> int:
    pr = d.probe
    if pr.ok:
        print("sub-stream decodes cleanly in an independent session; no fault found")
        return 0
    print(f"fault              AU {pr.fault_au} "
          f"(source decode {pr.fault_source_index}): {pr.error['error']}")
    print(f"frozen at          "
          f"{'no picture ever shown' if pr.frozen_at is None else pr.frozen_at}")
    broken = [row for row in d.view if row.missing_refs]
    if broken:
        print("missing references:")
        for row in broken:
            print(f"  AU {row.index} (source {row.source_index}, display "
                  f"{row.presentation}, tid {row.temporal_id}) needs source "
                  f"{row.missing_refs}")
    if not d.view[0].config_active:
        print("configuration      first AU has no active VPS/SPS/PPS")
    print(f"cause              {d.cause.value}")
    for e in d.experiments:
        if not e.applicable:
            print(f"  what-if {e.action}: not applicable ({e.detail.get('reason', '')})")
        else:
            print(f"  what-if {e.action}: "
                  f"{'RECOVERS' if e.recovered else 'still fails'}"
                  f"{'' if not e.recovered else f' (first clean display {e.first_clean})'}")
    if d.minimal is not None:
        m = d.minimal
        print(f"minimal reproducer {m.au_count} picture AU(s) "
              f"{m.source_indices}, {len(m.data)} bytes, still fails")
    if d.recovery_point is not None:
        rp = d.recovery_point
        note = " (needs parameter sets lent from the source)" \
            if rp.needs_parameter_sets else ""
        print(f"recovery point     display {rp.presentation} "
              f"(source decode {rp.source_index}, verified enterable){note}")
    else:
        print("recovery point     none before the fault AU")
    if d.repair is not None:
        r = d.repair
        print(f"repair             {r.kind}: {r.au_count} AUs, {len(r.data)} bytes, "
              f"added source AUs {r.added_source_indices}")
        print(f"first recovery     "
              f"{'none' if r.first_recovery is None else f'display {r.first_recovery}'}")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="layered_video.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("info", "plan", "verify", "extract"):
        p = sub.add_parser(name)
        p.add_argument("stream")
        if name != "info":
            p.add_argument("--at", type=int, required=True,
                           help="requested switch presentation index")
            p.add_argument("--layers", default="0,1,2",
                           help="temporal layer set after the switch (default 0,1,2)")
            p.add_argument("--low", default="0",
                           help="layers kept before the switch (default 0)")
            p.add_argument("--policy", default=None,
                           choices=[*[x.value for x in Policy], "forced"])
        if name == "extract":
            p.add_argument("--out", required=True)
        if name in ("plan", "verify"):
            p.add_argument("--json", action="store_true")

    d = sub.add_parser("diagnose",
                       help="locate the broken link of a sub-stream, per access unit")
    d.add_argument("stream", help="the (possibly broken) sub-stream")
    d.add_argument("--source", required=True,
                   help="source stream the sub-stream was extracted from")
    d.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    blob = open(args.stream, "rb").read()

    if args.cmd == "diagnose":
        source = open(args.source, "rb").read()
        d = diagnose(blob, source)
        if args.json:
            print(json.dumps(d.as_dict(), indent=2))
            return 0 if d.cause.value == "none" else 1
        return _print_diagnosis(d)

    layout = parse_stream(blob)

    if args.cmd == "info":
        _print_layout(layout)
        return 0

    layers = _layers(args.layers)
    low = _layers(args.low)
    policy = args.policy or ("preroll" if args.cmd == "extract" else "defer")

    analysis = analyse(blob, layout, args.at, layers, policy, low,
                       verify=args.cmd != "plan")
    summary = analysis.summary()
    if args.cmd == "plan" and args.json:
        print(json.dumps(summary, indent=2))
        return 0 if analysis.plan.is_legal else 2

    if args.cmd == "extract":
        if analysis.error:
            print(f"error: {analysis.error}", file=sys.stderr)
            return 2
        with open(args.out, "wb") as fh:
            fh.write(analysis.stream.data)
        print(f"wrote {args.out}: {analysis.stream.total_bytes} bytes, "
              f"{analysis.stream.access_unit_count} pictures "
              f"(layer set {layers}, policy {policy}, "
              f"switch at presentation {analysis.plan.switch_presentation})")
        return 0

    print(f"requested switch   presentation {summary['requested_presentation']}")
    print(f"policy             {policy} — {POLICY_LABELS.get(policy, policy)}")
    print(f"layer set          {layers}  (before switch: {low})")
    print(f"plan legal         {summary['legal']}")
    if summary["missing"]:
        print("missing references:")
        for m in summary["missing"]:
            print(f"  presentation {m['picture_display']} (tid "
                  f"{m['picture_temporal_id']}) needs presentation "
                  f"{m['missing_display']} (tid {m['missing_temporal_id']})")
    if analysis.error:
        print(f"extractor          {analysis.error}")
        return 2
    print(f"switch shown at    presentation {summary['switch_presentation']} "
          f"(deferred by {summary['deferred_by']})")
    if summary["preroll"]:
        print(f"pre-roll pictures  {summary['preroll']} "
              f"(decode indices kept for the exact switch)")
    if analysis.stream is not None:
        print(f"sub-stream         {summary['bytes_total']} bytes, "
              f"{summary['bytes_pictures']} picture bytes, "
              f"{summary['picture_count']} access units")
    if analysis.evaluation is None:
        return 0

    ev = analysis.evaluation
    print(f"clean decode       {len(ev.frames)} timeline slots, "
          f"{len(ev.decode_errors)} decoder error(s)")
    counts = summary["counts"]
    print(f"frames             recovered {counts['recovered']}, "
          f"base {counts['base']}, dropped {counts['missing']}, "
          f"duplicate {counts['duplicate']}, corrupt {counts['corrupt']}")
    if summary["enhancement_before_switch"]:
        print(f"enh before switch  {summary['enhancement_before_switch']} "
              f"(carried so the switch lands exactly at {summary['switch_presentation']})")
    rec = summary["first_recovery"]
    print(f"first recovery     "
          f"{'none' if rec is None else f'presentation {rec}'}")
    print()
    print(" presentation  tid  recovers  dB    verdict")
    for f in ev.frames:
        flag = "yes" if f.recovers else "no "
        print(f" {f.presentation:12d}  {f.temporal_id:3d}  {flag}      "
              f"{f.psnr:5.1f}  {f.reason}")
    return 0 if analysis.plan.is_legal else 2


if __name__ == "__main__":
    raise SystemExit(main())
