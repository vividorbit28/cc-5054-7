"""End-to-end checks over the generated four-layer clip.

Run with:  python -m unittest discover -s tests -v
(the clip is produced by ``python tools/make_sample.py``)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STREAM = ROOT / "data" / "sample_4layer.h265"

from layered_video import bitstream as bs  # noqa: E402
from layered_video.analysis import analyse  # noqa: E402
from layered_video.decode import (  # noqa: E402
    decode_reference, decode_substream, psnr,
)
from layered_video.extractor import (  # noqa: E402
    ExtractError, Policy, build_substream, extract, plan_extraction,
)

LAYERS = [0, 1, 2]
# Two requests with the *same* retained layer set.  At presentation 6 the
# immediate cut is missing enhancement references; at 10 it needs none.
GAPPED = 6
CLEAN = 10


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STREAM.exists():
            raise unittest.SkipTest(f"{STREAM} missing; run tools/make_sample.py")
        cls.blob = STREAM.read_bytes()
        cls.layout = bs.parse_stream(cls.blob)


class LayerTests(Base):
    def test_real_temporal_layers(self):
        self.assertEqual(self.layout.temporal_ids, [0, 1, 2, 3])
        for tid in self.layout.temporal_ids:
            self.assertGreater(
                sum(1 for p in self.layout.pictures if p.temporal_id == tid), 0)

    def test_layer_ids_read_from_nal_header(self):
        # TemporalId lives in the low three bits of the second NAL header byte
        # (stored as TemporalId+1), independently of the parser's bookkeeping.
        checked = 0
        for nal in self.layout.nals:
            if not nal.is_vcl:
                continue
            self.assertEqual(nal.temporal_id, (nal.data[1] & 0x07) - 1)
            checked += 1
        self.assertEqual(checked, self.layout.frame_count)

    def test_actual_references_are_resolved(self):
        inner = [p for p in self.layout.pictures if not p.is_irap]
        self.assertTrue(inner)
        self.assertTrue(any(p.ref_indices for p in inner))
        for p in self.layout.pictures:
            for ref in p.ref_indices:
                self.assertLess(ref, p.index)  # reference must precede its user
        # the first IDR needs nothing, and a gapped cut is caused by an
        # enhancement reference that the layer set keeps but a naive cut drops
        self.assertEqual(self.layout.pictures[0].ref_indices, [])


class PlanTests(Base):
    def test_same_layer_set_two_moments(self):
        gapped = plan_extraction(self.layout, GAPPED, LAYERS, Policy.NAIVE, [0])
        clean = plan_extraction(self.layout, CLEAN, LAYERS, Policy.NAIVE, [0])
        self.assertFalse(gapped.is_legal)
        self.assertTrue(clean.is_legal)
        # the gap is a dropped enhancement picture, not a base-layer anchor
        for m in gapped.missing:
            self.assertNotEqual(m.missing_temporal_id, 0)
        self.assertEqual(clean.preroll_decode_indices, [])

    def test_extractor_refuses_broken_cut(self):
        plan = plan_extraction(self.layout, GAPPED, LAYERS, Policy.NAIVE, [0])
        with self.assertRaises(ExtractError):
            build_substream(self.layout, self.blob, plan)

    def test_defer_moves_past_the_gap(self):
        plan = plan_extraction(self.layout, GAPPED, LAYERS, Policy.DEFER, [0])
        self.assertTrue(plan.is_legal)
        self.assertGreater(plan.switch_presentation, GAPPED)
        self.assertGreater(plan.deferred_by, 0)

    def test_preroll_keeps_switch_but_adds_references(self):
        gapped = plan_extraction(self.layout, GAPPED, LAYERS, Policy.PREROLL, [0])
        clean = plan_extraction(self.layout, CLEAN, LAYERS, Policy.PREROLL, [0])
        self.assertEqual(gapped.switch_presentation, GAPPED)
        self.assertTrue(gapped.preroll_decode_indices)
        self.assertEqual(clean.preroll_decode_indices, [])
        self.assertGreater(gapped.kept_count, clean.kept_count)


class DecodeTests(Base):
    def test_strict_decoder_rejects_missing_reference(self):
        plan = plan_extraction(self.layout, GAPPED, LAYERS, Policy.NAIVE, [0])
        with self.assertRaises(ExtractError):
            build_substream(self.layout, self.blob, plan)
        # bypass the guard to prove the decoder itself rejects the stream
        saved, plan.missing = plan.missing, []
        try:
            broken = build_substream(self.layout, self.blob, plan)
        finally:
            plan.missing = saved
        sub = bs.parse_stream(broken.data)
        result = decode_substream(broken.data, layout=sub,
                                  presentations=broken.presentations,
                                  strict=True)
        self.assertTrue(result.errors)
        self.assertIsNotNone(result.refused_at_decode_index)

    def test_substream_is_bit_exact_at_reference_quality(self):
        _, stream = extract(self.blob, CLEAN, LAYERS, Policy.PREROLL, [0])
        sub = bs.parse_stream(stream.data)
        decoded = decode_substream(stream.data, layout=sub,
                                   presentations=stream.presentations)
        reference = decode_reference(self.blob, self.layout)
        self.assertFalse(decoded.errors)
        checked = 0
        for pic in decoded.pictures:
            ref = reference.get(pic.presentation)
            if ref is None:
                continue
            self.assertGreater(psnr(pic.yuv, ref.yuv), 60.0)
            checked += 1
        self.assertGreater(checked, 40)


class AnalysisTests(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.reference = decode_reference(cls.blob, cls.layout)

    def _run(self, at, policy):
        return analyse(self.blob, self.layout, at, LAYERS, policy, [0],
                       reference=self.reference)

    def test_first_recovery_is_first_kept_enhancement_frame(self):
        a = self._run(GAPPED, "preroll")
        self.assertTrue(a.ok)
        self.assertEqual(a.evaluation.first_recovery, GAPPED)
        b = self._run(GAPPED, "defer")
        self.assertEqual(b.evaluation.first_recovery, b.plan.switch_presentation)
        self.assertGreaterEqual(b.evaluation.first_recovery, CLEAN)

    def test_deferred_switch_never_reports_earlier_recovery(self):
        a = self._run(GAPPED, "defer")
        self.assertEqual(a.summary()["deferred_by"],
                         a.plan.switch_presentation - GAPPED)
        for f in a.evaluation.frames:
            if f.recovers:
                self.assertGreaterEqual(f.presentation, GAPPED)

    def test_forced_cut_never_counts_as_recovery(self):
        a = self._run(GAPPED, "forced")
        self.assertFalse(a.plan.is_legal)
        self.assertIsNone(a.summary()["first_recovery"])
        self.assertEqual(a.summary()["counts"]["recovered"], 0)
        self.assertTrue(all("illegal" in f.reason for f in a.evaluation.frames
                            if f.decoded and
                            f.temporal_id in set(LAYERS) - {0}))

    def test_byte_accounting_matches_stream(self):
        a = self._run(GAPPED, "preroll")
        s = a.summary()
        self.assertEqual(s["bytes_total"], len(a.stream.data))
        self.assertEqual(s["picture_count"], a.stream.access_unit_count)
        self.assertEqual(s["bytes_pictures"] + s["bytes_parameters"],
                         s["bytes_total"])


if __name__ == "__main__":
    unittest.main()
