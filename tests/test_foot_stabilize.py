"""Closed-loop planted-foot stabilization: lock weight map shape."""

import unittest

from mocap_doctor.core import target


class SegmentLockWeightTests(unittest.TestCase):
    def test_interior_frames_take_the_full_correction(self):
        weights = target._segment_lock_weights(10, 20, 2, 1, 100)
        for frame in range(10, 21):
            self.assertEqual(weights[frame], 1.0)

    def test_the_blend_ring_eases_to_zero(self):
        """Ring distance k of b blends at (b + 1 - k)/(b + 1), like the locks."""

        weights = target._segment_lock_weights(10, 20, 2, 1, 100)
        self.assertAlmostEqual(weights[9], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(weights[8], 1.0 / 3.0, places=9)
        self.assertAlmostEqual(weights[21], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(weights[22], 1.0 / 3.0, places=9)
        self.assertNotIn(7, weights)
        self.assertNotIn(23, weights)

    def test_no_blend_means_no_ring(self):
        weights = target._segment_lock_weights(10, 20, 0, 1, 100)
        self.assertEqual(
            sorted(weights), list(range(10, 21))
        )

    def test_the_ring_is_clipped_to_the_frame_range(self):
        weights = target._segment_lock_weights(1, 6, 3, 1, 6)
        self.assertEqual(min(weights), 1)
        self.assertEqual(max(weights), 6)

    def test_single_frame_segments_are_lockable(self):
        weights = target._segment_lock_weights(5, 5, 1, 1, 10)
        self.assertEqual(weights[5], 1.0)
        self.assertAlmostEqual(weights[4], 0.5, places=9)
        self.assertAlmostEqual(weights[6], 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
