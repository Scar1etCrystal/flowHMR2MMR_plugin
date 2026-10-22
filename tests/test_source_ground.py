"""Ground estimation for the source-side floor leveling."""

import unittest

from mocap_doctor.core import source


class RollingLowPercentileTests(unittest.TestCase):
    def test_flat_ground_stays_flat(self):
        values = {f: 0.0 for f in range(1, 51)}
        out = source._rolling_low_percentile(values, 1, 50, 10)
        self.assertEqual(set(out.values()), {0.0})

    def test_a_brief_lift_does_not_move_the_ground(self):
        """A jump puts both feet up; the ground under it has not moved."""

        values = {f: 0.0 for f in range(1, 101)}
        for f in range(45, 56):
            values[f] = 0.5
        out = source._rolling_low_percentile(values, 1, 100, 10)
        self.assertLess(max(out[f] for f in range(45, 56)), 0.05)

    def test_a_slow_drift_is_followed(self):
        values = {f: f * 0.001 for f in range(1, 301)}
        out = source._rolling_low_percentile(values, 1, 300, 30)
        # A percentile, not a minimum: the window near frame 1 still contains
        # a few frames above it, so the estimate sits slightly high.
        self.assertLess(out[1], 0.01)
        self.assertGreater(out[300], out[1] + 0.25)

    def test_missing_samples_are_filled_from_the_nearest_estimate(self):
        values = {f: 0.1 for f in range(1, 21)}
        values[10] = None
        out = source._rolling_low_percentile(values, 1, 20, 5)
        self.assertNotIn(None, out.values())
        self.assertAlmostEqual(out[10], 0.1, places=6)

    def test_all_missing_returns_none(self):
        values = {f: None for f in range(1, 11)}
        self.assertEqual(set(source._rolling_low_percentile(values, 1, 10, 3).values()), {None})


if __name__ == "__main__":
    unittest.main()
