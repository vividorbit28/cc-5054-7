"""Broken-chain locator checks: two frozen short sequences, one root-caused to
lost parameter sets, the other to a cropped enhancement reference.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STREAM = ROOT / "data" / "sample_4layer.h265"

from layered_video import locator as lc  # noqa: E402
from layered_video.bitstream import NAL_PPS, NAL_SPS, NAL_VPS, parse_stream  # noqa: E402
from layered_video.decode import decode_reference  # noqa: E402
from layered_video.extractor import _nal_bytes, _picture_bytes, _prefix_nals  # noqa: E402

LAYERS = {0, 1, 2}
# The cropped enhancement picture: source decode 35 (display 34, tid 2) is
# referenced by source decode 38 (display 38, tid 2) inside the layer set.
DROPPED_REF = 35
BROKEN_AT = 38


def _au_bytes(layout, blob, picture, drop_param_types=()):
    out = b"".join(_nal_bytes(n) for n in _prefix_nals(layout, picture)
                   if n.nal_type not in drop_param_types)
    return out + _picture_bytes(layout, blob, picture)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STREAM.exists():
            raise unittest.SkipTest(f"{STREAM} missing; run tools/make_sample.py")
        cls.blob = STREAM.read_bytes()
        cls.layout = parse_stream(cls.blob)
        cls.reference = decode_reference(cls.blob, cls.layout)
        # Two short sequences over the same span, both carrying layers 0,1,2.
        cls.src_aus = [p for p in cls.layout.pictures
                       if 32 <= p.display_index < 96 and p.temporal_id <= 2]
        # Sequence A: the cut only becomes decodable after the parameter-set
        # update repeated at the next IDR — the head VPS/SPS/PPS were lost at
        # extraction time.
        cls.seq_a = (
            _au_bytes(cls.layout, cls.blob, cls.src_aus[0],
                      drop_param_types=(NAL_VPS, NAL_SPS, NAL_PPS))
            + b"".join(_au_bytes(cls.layout, cls.blob, p)
                       for p in cls.src_aus[1:]))
        # Sequence B: starts at a proper random access point (IDR at display
        # 32) with configuration intact, but one enhancement picture was
        # cropped away.
        cls.seq_b = b"".join(_au_bytes(cls.layout, cls.blob, p)
                             for p in cls.src_aus if p.index != DROPPED_REF)
        # Sequence C: a legal excerpt; repeats VPS/SPS/PPS at the second IDR.
        cls.seq_c = b"".join(_au_bytes(cls.layout, cls.blob, p)
                             for p in cls.src_aus)

    def _aus(self, seq):
        aus = lc.split_access_units(seq)
        lc.align_to_source(aus, self.layout)
        return aus


class FaultSequenceTests(Base):
    def test_both_sequences_carry_the_same_layers(self):
        for seq in (self.seq_a, self.seq_b):
            aus = self._aus(seq)
            tids = {au.temporal_id for au in aus if au.temporal_id is not None}
            self.assertEqual(tids, LAYERS)

    def test_both_sequences_freeze(self):
        # Same surface symptom: a strict independent decoder refuses the
        # stream and playback has no picture to show.
        for seq in (self.seq_a, self.seq_b):
            pr = lc.probe(self._aus(seq))
            self.assertFalse(pr.ok)
            self.assertEqual(pr.emitted, [])
            self.assertIsNone(pr.frozen_at)


class DependencyViewTests(Base):
    def test_sequence_a_lacks_config_but_no_reference(self):
        view = lc.dependency_view(self._aus(self.seq_a))
        self.assertFalse(view[0].config_active)
        self.assertEqual(view[0].parameter_sets, [])
        self.assertTrue(all(not row.missing_refs for row in view))
        # the parameter sets only reappear at the next IDR (display 64)
        carriers = [row for row in view if row.parameter_sets]
        self.assertTrue(carriers)
        self.assertEqual(carriers[0].presentation, 64)

    def test_sequence_b_lacks_a_reference_but_has_config(self):
        view = lc.dependency_view(self._aus(self.seq_b))
        self.assertTrue(view[0].config_active)
        broken = [row for row in view if row.missing_refs]
        self.assertEqual(len(broken), 1)
        self.assertEqual(broken[0].source_index, BROKEN_AT)
        self.assertEqual(broken[0].missing_refs, [DROPPED_REF])


class WhatIfTests(Base):
    def test_sequence_a_recovers_only_by_swapping_parameter_sets(self):
        aus = self._aus(self.seq_a)
        swap = lc.what_if_replace_parameter_sets(aus, self.layout, self.reference)
        restore = lc.what_if_restore_reference(aus, self.layout, self.reference)
        self.assertTrue(swap.applicable)
        self.assertTrue(swap.recovered)
        self.assertEqual(swap.first_clean, 32)
        # nothing references a cropped frame, so there is no reference to restore
        self.assertFalse(restore.applicable)
        self.assertFalse(restore.recovered)

    def test_sequence_b_recovers_only_by_restoring_one_reference(self):
        aus = self._aus(self.seq_b)
        swap = lc.what_if_replace_parameter_sets(aus, self.layout, self.reference)
        restore = lc.what_if_restore_reference(aus, self.layout, self.reference)
        # configuration was never the problem: swapping it changes nothing
        self.assertTrue(swap.applicable)
        self.assertFalse(swap.recovered)
        self.assertTrue(restore.applicable)
        self.assertEqual(restore.detail["restored_source_index"], DROPPED_REF)
        self.assertTrue(restore.recovered)
        self.assertEqual(restore.first_clean, 32)


class MinimizerTests(Base):
    def test_sequence_a_minimises_to_one_unconfigured_picture(self):
        aus = self._aus(self.seq_a)
        minimal = lc.minimize_failure(aus)
        self.assertEqual(minimal.au_count, 1)
        self.assertEqual(minimal.source_indices, [32])
        self.assertFalse(minimal.probe.ok)
        self.assertEqual(minimal.probe.fault_source_index, 32)
        # no parameter sets survive: they were never the trigger's remedy
        self.assertTrue(all(not au.parameter_types for au in minimal.aus))

    def test_sequence_b_minimises_keeping_the_same_fault(self):
        aus = self._aus(self.seq_b)
        minimal = lc.minimize_failure(aus)
        original = sum(1 for au in aus if au.vcl_nals)
        self.assertLess(minimal.au_count, original)
        self.assertNotIn(DROPPED_REF, minimal.source_indices)
        self.assertFalse(minimal.probe.ok)
        # the same source picture still fails, with configuration active and
        # the reference still absent (the fault did not mutate into lost config)
        self.assertEqual(minimal.probe.fault_source_index, BROKEN_AT)
        self.assertEqual(minimal.signature[0], BROKEN_AT)
        self.assertTrue(minimal.signature[2])   # config was active at the fault
        self.assertTrue(minimal.signature[3])   # reference still missing


class RepairTests(Base):
    def test_sequence_a_repair_is_an_extractor_fix(self):
        d = lc.diagnose(self.seq_a, self.blob,
                        layout=self.layout, reference=self.reference)
        self.assertEqual(d.cause, lc.Cause.MISSING_PARAMETER_SETS)
        self.assertEqual(d.repair.kind, "extractor")
        self.assertEqual(d.repair.added_source_indices, [])
        # only parameter-set bytes were added; no access unit appeared
        self.assertEqual(d.repair.au_count, len(self.src_aus))
        self.assertTrue(d.repair.recovered)
        self.assertEqual(d.repair.first_recovery, 32)

    def test_sequence_b_repair_is_a_capability_fix(self):
        d = lc.diagnose(self.seq_b, self.blob,
                        layout=self.layout, reference=self.reference)
        self.assertEqual(d.cause, lc.Cause.MISSING_REFERENCE)
        self.assertEqual(d.repair.kind, "capability")
        self.assertEqual(d.repair.added_source_indices, [DROPPED_REF])
        self.assertEqual(d.repair.au_count, len(self.src_aus))
        self.assertTrue(d.repair.recovered)
        self.assertEqual(d.repair.first_recovery, 32)

    def test_repair_never_pastes_the_whole_source_back(self):
        for seq in (self.seq_a, self.seq_b):
            d = lc.diagnose(seq, self.blob,
                            layout=self.layout, reference=self.reference)
            self.assertTrue(d.repair.recovered)
            self.assertLess(d.repair.au_count, self.layout.frame_count)
            self.assertLess(len(d.repair.data), len(self.blob))
            self.assertNotEqual(d.repair.data, self.blob)


class LegalExcerptTests(Base):
    def test_sequence_c_fully_recovers(self):
        aus = self._aus(self.seq_c)
        verdict = lc.judge(aus, self.reference)
        self.assertTrue(verdict.recovered)
        self.assertEqual(verdict.first_clean, 32)
        self.assertEqual(len(verdict.probe.emitted), len(self.src_aus))

    def test_redundant_parameter_sets_are_not_required(self):
        aus = self._aus(self.seq_c)
        report = lc.required_segments(aus, self.reference)
        prefixes = {r.au_index: r for r in report if r.kind == "prefix"
                    and "SPS" in r.label}
        # the head VPS/SPS/PPS are required; the repeat at the second IDR
        # (display 64) decodes fine without and must not be reported required
        head = next(r for r in prefixes.values() if r.au_index == 0)
        repeated = next(r for r in prefixes.values() if r.au_index != 0)
        self.assertTrue(head.required)
        self.assertFalse(repeated.required)
        self.assertEqual(aus[repeated.au_index].presentation, 64)

    def test_pictures_with_dependents_are_required(self):
        aus = self._aus(self.seq_c)
        report = lc.required_segments(aus, self.reference)
        pictures = {r.source_index: r for r in report if r.kind == "picture"}
        # the IDR anchor and the P frames are referenced by kept pictures
        self.assertTrue(pictures[32].required)
        self.assertTrue(pictures[33].required)
        # display 38 (tid 2) is only referenced from tid-3 pictures, which
        # this layer set does not carry: dropping it breaks nothing
        self.assertFalse(pictures[BROKEN_AT].required)


class RecoveryPointTests(Base):
    def test_sequence_b_recovery_point_is_the_previous_idr(self):
        aus = self._aus(self.seq_b)
        pr = lc.probe(aus)
        point = lc.nearest_recovery_point(aus, pr.fault_au,
                                          self.layout, self.reference)
        self.assertIsNotNone(point)
        self.assertEqual(point.presentation, 32)
        self.assertEqual(point.source_index, 32)
        self.assertFalse(point.needs_parameter_sets)
        self.assertTrue(point.verified)

    def test_sequence_a_has_no_enterable_picture_before_the_fault(self):
        aus = self._aus(self.seq_a)
        pr = lc.probe(aus)
        self.assertEqual(pr.fault_au, 0)
        point = lc.nearest_recovery_point(aus, pr.fault_au,
                                          self.layout, self.reference)
        self.assertIsNone(point)


class DiagnoseTests(Base):
    def test_legal_stream_reports_no_fault(self):
        d = lc.diagnose(self.seq_c, self.blob,
                        layout=self.layout, reference=self.reference)
        self.assertEqual(d.cause, lc.Cause.NONE)
        self.assertIsNone(d.repair)

    def test_diagnosis_view_tracks_every_access_unit(self):
        d = lc.diagnose(self.seq_b, self.blob,
                        layout=self.layout, reference=self.reference)
        pictures = [row for row in d.view if row.source_index is not None]
        self.assertEqual(len(pictures), len(self.src_aus) - 1)
        self.assertNotIn(DROPPED_REF, [row.source_index for row in pictures])


if __name__ == "__main__":
    unittest.main()
