"""Generate the demo clip with *real* HEVC temporal sub-layers.

x265 only emits temporal sub-layers when asked for at least two of them
(``temporal-layers=1`` is silently disabled), so this script asks for four and
then verifies from the NAL headers that TemporalId values 0..3 actually exist.
"""
from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layered_video.bitstream import parse_stream  # noqa: E402

W, H, FPS = 480, 270, 24
N_FRAMES = 96
KEYINT = 32
CRF = 12
TEMPORAL_LAYERS = 4


def make_frames(n: int):
    """High-entropy still texture plus a *very slowly* moving subject.

    Slow motion keeps the inter-coded (enhancement) layers cheap, while dense
    texture makes the coarse base layer visibly soft.  That is what lets the
    verifier tell "target quality reached" apart from "decoder duplicated the
    previous frame".
    """
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:H, 0:W]
    # Fine luminance/chroma texture: ~uniform, uncorrelated noise.
    lum = rng.integers(16, 240, (H, W)).astype(np.int16)
    # Chroma must be smooth: 4:2:0 subsampling would destroy per-pixel noise,
    # and the verifier compares decoded RGB against this source.
    chroma = np.stack([
        128 + 70 * np.sin(xx / 53.0 + yy / 71.0),
        128 + 70 * np.cos(xx / 61.0 - yy / 47.0),
    ]).astype(np.int16)
    # A soft smooth plate so the coarse layer still looks plausible.
    plate = 96 + 60 * np.sin(xx / 37.0) * np.cos(yy / 29.0)
    out = []
    for i in range(n):
        # Y = fine texture (the part a coarse QP destroys), U/V smooth.
        y = np.clip(0.85 * lum + 0.15 * plate, 0, 255).astype(np.uint8)
        u = np.clip(chroma[0], 0, 255).astype(np.uint8)
        v = np.clip(chroma[1], 0, 255).astype(np.uint8)
        # slowly drifting subject plus a phase-shifted smooth band, so no two
        # source frames are identical (the verifier relies on that)
        x = int((i * 4) % (W - 80))
        sy = int((H // 2) + 40 * np.sin(i / 23.0))
        sy = max(10, min(H - 70, sy))
        y[sy:sy + 60, x:x + 80] = 235
        u[sy:sy + 60, x:x + 80] = 90
        v[sy:sy + 60, x:x + 80] = 200
        band = (40 * np.sin((xx + i * 3) / 11.0)).astype(np.int16)
        y = np.clip(y.astype(np.int16) + band, 0, 255).astype(np.uint8)
        # frame index drawn as bright bars, changes once per frame
        for b in range(9):
            if (i >> b) & 1:
                y[6:26, 6 + b * 20: 22 + b * 20] = 255
        out.append(np.stack([y, u, v]))
    return out


def _encode_once(path: Path, frames, temporal_layers: int, extra: str = "") -> None:
    params = (
        f"bframes=7:b-pyramid=1:temporal-layers={temporal_layers}"
        f":keyint={KEYINT}:min-keyint={KEYINT}:scenecut=0:open-gop=0"
        ":repeat-headers=1:aud=1:frame-threads=1:pools=1:log-level=error"
    )
    if extra:
        params += ":" + extra
    with av.open(str(path), "w", format="hevc") as oc:
        st = oc.add_stream("libx265", rate=FPS, options={
            "preset": "medium", "x265-params": params})
        st.width, st.height, st.pix_fmt = W, H, "yuv444p"
        for i, a in enumerate(frames):
            f = av.VideoFrame.from_ndarray(a, format="yuv444p")
            f.pts = i
            f.time_base = Fraction(1, FPS)
            for pkt in st.encode(f):
                oc.mux(pkt)
        for pkt in st.encode():
            oc.mux(pkt)


# The base layer is coded coarsely; enhancement layers are nearly transparent.
# "Target quality" for the verifier is bit-exactness with a clean decode of the
# full stream, so the exact QP values only need to make the base layer
# measurably softer than the enhancement layers.
LAYER_QP = {0: 51, 1: 34, 2: 24, 3: 16}


def encode(path: Path, frames, temporal_layers: int) -> None:
    """Two-pass encode: learn the layer ids, then re-encode with per-layer QP."""
    tmp = path.with_suffix(".probe.h265")
    _encode_once(tmp, frames, temporal_layers)
    probe = parse_stream(tmp.read_bytes())
    tid_by_display = {p.display_index: p.temporal_id for p in probe.pictures}
    tmp.unlink()
    zones = "/".join(
        f"{d},{d},q={LAYER_QP[tid_by_display[d]]}" for d in range(len(frames)))
    _encode_once(path, frames, temporal_layers, extra=f"zones={zones}")
    final = parse_stream(path.read_bytes())
    if {p.display_index: p.temporal_id for p in final.pictures} != tid_by_display:
        raise RuntimeError("layer structure changed between probe and final encode")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_4layer.h265")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--check", action="store_true", help="only re-check an existing file")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.check:
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = make_frames(args.frames)
        encode(out, frames, TEMPORAL_LAYERS)
        # Save the raw source so the verifier can measure quality against a
        # reference that no decoder produced.
        # Full RGB source frames: the verifier compares both luma and chroma
        # and needs adjacent frames to differ, so a luma-only save is not enough.
        src = out.with_name("source_frames.npz")
        np.savez_compressed(src, yuv=np.stack(frames))
        print(f"source frames -> {src}")
    layout = parse_stream(out.read_bytes())
    tids = layout.temporal_ids
    if len(tids) < 2 or tids != list(range(len(tids))):
        print(f"ERROR: encoder did not emit real temporal layers (ids={tids})", file=sys.stderr)
        return 1
    print(f"{out}: {layout.frame_count} pictures, {layout.gop_count} GOPs, "
          f"temporal ids {tids}, {out.stat().st_size} bytes")
    counts = {t: sum(1 for p in layout.pictures if p.temporal_id == t) for t in tids}
    print("pictures per temporal id:", counts)
    display_pattern = [p.temporal_id for p in sorted(layout.pictures, key=lambda p: p.display_index)]
    print("presentation-order temporal ids:", display_pattern[:40], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
