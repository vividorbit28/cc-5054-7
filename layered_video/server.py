"""Local preview server for the layered sub-stream tool.

The browser cannot decode HEVC, so each verified sub-stream is also transcoded
to H.264 for *playback only*.  Every number shown in the UI comes from the
strict server-side decode of the original HEVC sub-stream, never from the
H.264 preview proxy.

    python -m layered_video.server --stream data/sample_4layer.h265
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import traceback
from dataclasses import dataclass
from fractions import Fraction
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import av
import numpy as np

from .analysis import analyse
from .bitstream import StreamLayout, parse_stream
from .decode import DecodedPicture, decode_reference
from .extractor import Policy

FPS = 24
STATIC = Path(__file__).resolve().parent / "static"

STATE: "ServerState | None" = None
BUILD_LOCK = threading.Lock()


@dataclass
class Build:
    key: str
    summary: dict
    frames: list[dict]
    media: bytes
    duration_ms: int


class ServerState:
    def __init__(self, stream_path: Path):
        self.path = stream_path
        self.blob = stream_path.read_bytes()
        self.layout: StreamLayout = parse_stream(self.blob)
        self.reference: dict[int, DecodedPicture] = decode_reference(
            self.blob, self.layout)
        self.builds: dict[str, Build] = {}
        self.order: list[str] = []
        self._lock = threading.Lock()

    def info(self) -> dict:
        layout = self.layout
        pictures = []
        for p in sorted(layout.pictures, key=lambda p: p.display_index):
            pictures.append({
                "presentation": p.display_index,
                "decode_index": p.index,
                "temporal_id": p.temporal_id,
                "type": p.picture_type,
                "gop": p.gop,
                "references": [layout.pictures[r].display_index
                               for r in p.ref_indices],
            })
        return {
            "name": self.path.name,
            "frame_count": layout.frame_count,
            "gop_count": layout.gop_count,
            "temporal_ids": layout.temporal_ids,
            "fps": FPS,
            "full_bytes": len(self.blob),
            "pictures": pictures,
            "layer_counts": {
                str(t): sum(1 for p in layout.pictures if p.temporal_id == t)
                for t in layout.temporal_ids},
        }

    def build(self, at: int, layers: list[int], low: list[int],
              policy: str, target_psnr: float) -> Build:
        descriptor = (f"{at}|{','.join(map(str, layers))}|"
                      f"{','.join(map(str, low))}|{policy}|{target_psnr}")
        key = hashlib.sha1(descriptor.encode()).hexdigest()[:16]
        with self._lock:
            cached = self.builds.get(key)
            if cached is not None:
                return cached

        with BUILD_LOCK:
            analysis = analyse(self.blob, self.layout, at, layers, policy, low,
                               reference=self.reference, verify=True)
            summary = analysis.summary()
            frames = analysis.frames()
            if analysis.evaluation is not None:
                summary["target_psnr"] = analysis.evaluation.target_psnr
            elif "target_psnr" not in summary:
                summary["target_psnr"] = target_psnr

            if analysis.stream is not None:
                decoded = _decode_for_preview(analysis.stream, self.layout)
                media = _encode_mp4(decoded, self.layout.frame_count)
            else:
                media = b""
            duration = int(round(self.layout.frame_count * 1000 / FPS))
            summary["duration_ms"] = duration
            summary["fps"] = FPS
            build = Build(key=key, summary=summary, frames=frames,
                          media=media, duration_ms=duration)

        with self._lock:
            self.builds[key] = build
            self.order.append(key)
            while len(self.order) > 24:
                drop = self.order.pop(0)
                self.builds.pop(drop, None)
        return build


def _decode_for_preview(stream, layout: StreamLayout) -> list[DecodedPicture]:
    from .decode import decode_substream
    sub = parse_stream(stream.data)
    decoded = decode_substream(stream.data, layout=sub,
                               presentations=stream.presentations,
                               strict=False)
    return decoded.pictures


def _encode_mp4(pictures: list[DecodedPicture], frame_count: int) -> bytes:
    if not pictures:
        return b""
    buf = io.BytesIO()
    h, w = pictures[0].yuv.shape[1], pictures[0].yuv.shape[2]
    with av.open(buf, "w", format="mp4") as oc:
        st = oc.add_stream("libx264", rate=FPS, options={
            "preset": "veryfast", "crf": "18", "g": "4", "bf": "0",
            "pix_fmt": "yuv420p"})
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.time_base = Fraction(1, 1000)
        for pic in pictures:
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(pic.yuv), format="yuv444p")
            frame = frame.reformat(format="yuv420p")
            frame.pts = int(round(pic.presentation * 1000 / FPS))
            frame.time_base = Fraction(1, 1000)
            for packet in st.encode(frame):
                oc.mux(packet)
        for packet in st.encode():
            oc.mux(packet)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "LayeredPreview/1.0"

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
            return
        if path == "/api/info":
            self._json(STATE.info())
            return
        if path == "/api/build":
            self._handle_build(parse_qs(parsed.query))
            return
        if path.startswith("/media/"):
            key = path[len("/media/"):]
            if key.endswith(".mp4"):
                key = key[:-4]
            build = STATE.builds.get(key)
            if build is None:
                self._json({"error": "unknown build"}, 404)
                return
            self._send(200, build.media, "video/mp4")
            return
        if path == "/api/raw":
            # The exact HEVC sub-stream, for anyone who wants to decode it
            # themselves; the browser cannot play it.
            self._handle_raw(parse_qs(parsed.query))
            return
        self._json({"error": "not found"}, 404)

    def _parse_common(self, query: dict):
        at = int(query.get("at", ["0"])[0])
        layers = sorted({int(x) for x in query.get("layers", ["0,1,2"])[0]
                         .split(",") if x.strip() != ""})
        low = sorted({int(x) for x in query.get("low", ["0"])[0]
                      .split(",") if x.strip() != ""})
        policy = query.get("policy", ["defer"])[0]
        target = float(query.get("target", ["38"])[0])
        return at, layers, low, policy, target

    def _handle_build(self, query: dict):
        try:
            at, layers, low, policy, target = self._parse_common(query)
            if policy not in {p.value for p in Policy} | {"forced"}:
                raise ValueError(f"unknown policy {policy!r}")
            build = STATE.build(at, layers, low, policy, target)
        except Exception as exc:  # surfaced to the UI instead of a stack trace
            traceback.print_exc()
            self._json({"error": str(exc)}, 400)
            return
        payload = dict(build.summary)
        payload["frames"] = build.frames
        payload["media_url"] = f"/media/{build.key}.mp4"
        payload["raw_url"] = (f"/api/raw?at={at}&layers={','.join(map(str, layers))}"
                              f"&low={','.join(map(str, low))}&policy={policy}")
        self._json(payload)

    def _handle_raw(self, query: dict):
        try:
            at, layers, low, policy, _ = self._parse_common(query)
            analysis = analyse(STATE.blob, STATE.layout, at, layers, policy, low,
                               reference=STATE.reference, verify=False)
            if analysis.stream is None:
                self._json({"error": analysis.error or "cannot extract"}, 409)
                return
            filename = f"layers_{'-'.join(map(str, layers))}_at_{at}_{policy}.h265"
            self.send_response(200)
            self.send_header("Content-Type", "video/h265")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(analysis.stream.data)))
            self.end_headers()
            self.wfile.write(analysis.stream.data)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="layered_video.server")
    ap.add_argument("--stream", default="data/sample_4layer.h265")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    global STATE
    STATE = ServerState(Path(args.stream))
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {args.stream} on http://{args.host}:{args.port}/")
    print(f"  {STATE.layout.frame_count} pictures, temporal ids "
          f"{STATE.layout.temporal_ids}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
