"""Grounding: unconditional pin, ballistic airborne spans, per-stretch dynamics."""

import unittest

from mocap_doctor.core import target


class PinCorrectionTests(unittest.TestCase):
    def test_every_grounded_frame_is_pinned(self):
        lowest = {frame: 0.05 for frame in range(1, 6)}
        corrections, missing = target._pin_corrections(lowest, 1, 5, 0.01)
        self.assertEqual(set(corrections.values()), {0.01 - 0.05})
        self.assertEqual(missing, [])

    def test_the_correction_follows_the_foot(self):
        lowest = {1: 0.05, 2: 0.02, 3: 0.09}
        corrections, _missing = target._pin_corrections(lowest, 1, 3, 0.01)
        self.assertAlmostEqual(corrections[1], -0.04, places=6)
        self.assertAlmostEqual(corrections[2], -0.01, places=6)
        self.assertAlmostEqual(corrections[3], -0.08, places=6)

    def test_a_missing_sample_holds_the_previous_value(self):
        """A missing sample is a data gap, not a reason to move the body."""

        lowest = {1: 0.05, 2: None, 3: 0.02}
        corrections, missing = target._pin_corrections(lowest, 1, 3, 0.01)
        self.assertAlmostEqual(corrections[2], corrections[1], places=9)
        self.assertEqual(missing, [2])

    def test_a_leading_missing_sample_holds_zero(self):
        corrections, _missing = target._pin_corrections({1: None, 2: 0.05}, 1, 2, 0.01)
        self.assertEqual(corrections[1], 0.0)
        self.assertAlmostEqual(corrections[2], -0.04, places=6)


class BallisticTests(unittest.TestCase):
    """The arc is physics: gravity plus the marked flight time decide the apex."""

    def test_equal_anchors_give_the_physical_apex(self):
        fps = 30.0
        arc = target._ballistic_z(0.0, 0.0, 0, 18, [9], fps, target.GRAVITY)
        flight = 18 / fps
        self.assertAlmostEqual(arc[9], target.GRAVITY * flight**2 / 8, places=6)

    def test_the_arc_passes_through_both_anchors(self):
        arc = target._ballistic_z(
            0.2, 0.5, 10, 34, [10, 22, 34], 30.0, target.GRAVITY
        )
        self.assertAlmostEqual(arc[10], 0.2, places=6)
        self.assertAlmostEqual(arc[34], 0.5, places=6)

    def test_a_crushed_arc_is_restored(self):
        """0.6 s of flight implies a 44 cm apex whatever the data claims."""

        arc = target._ballistic_z(
            0.0, 0.0, 0, 18, list(range(0, 19)), 30.0, target.GRAVITY
        )
        self.assertGreater(max(arc.values()), 0.40)

    def test_a_longer_flight_flies_higher(self):
        short = target._ballistic_z(0.0, 0.0, 0, 12, [6], 30.0, target.GRAVITY)
        long = target._ballistic_z(0.0, 0.0, 0, 24, [12], 30.0, target.GRAVITY)
        self.assertGreater(long[12], short[6] * 3.5)


class GroundedDynamicsTests(unittest.TestCase):
    def test_the_rate_limit_applies_within_a_stretch(self):
        corrections = {frame: 0.0 for frame in range(1, 21)}
        corrections.update({frame: -0.20 for frame in range(10, 21)})
        out = target._smooth_limit_grounded(corrections, set(), 1, 20, 0, 0.01)
        steps = [abs(out[f + 1] - out[f]) for f in range(1, 20)]
        self.assertLessEqual(max(steps), 0.01 + 1e-9)

    def test_airborne_frames_are_never_touched(self):
        corrections = {frame: 0.0 for frame in range(1, 11)}
        out = target._smooth_limit_grounded(corrections, {4, 5, 6}, 1, 10, 2, 0.01)
        for frame in (4, 5, 6):
            self.assertEqual(out[frame], 0.0)

    def test_dynamics_do_not_bleed_across_a_flight(self):
        corrections = {frame: 0.0 for frame in range(1, 11)}
        corrections.update({frame: -0.05 for frame in range(1, 4)})
        out = target._smooth_limit_grounded(corrections, {4, 5, 6}, 1, 10, 3, 0.0)
        # The far stretch must not see the near one through the gap.
        self.assertEqual(out[7], 0.0)
        self.assertEqual(out[10], 0.0)

    def test_a_takeoff_keeps_its_speed(self):
        """The limit must not flatten a launch: airborne values stay untouched."""

        corrections = {frame: -0.05 for frame in range(1, 6)}
        corrections.update({frame: 0.30 for frame in range(6, 9)})  # the flight
        out = target._smooth_limit_grounded(corrections, {6, 7, 8}, 1, 9, 2, 0.01)
        self.assertEqual(out[6], 0.30)
        self.assertEqual(out[8], 0.30)


if __name__ == "__main__":
    unittest.main()
