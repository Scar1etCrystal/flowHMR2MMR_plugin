"""Unit tests for core/agent_fx.py - pure numpy write-tool math."""
import unittest

import numpy as np

from mocap_doctor.core import agent_fx


class DeltaTests(unittest.TestCase):
    def test_quat_delta_roundtrip(self):
        n = 10
        ang = np.radians(np.linspace(0, 30, n))
        base = np.stack([np.cos(ang / 2), np.zeros(n), np.sin(ang / 2),
                         np.zeros(n)], axis=1)
        extra = np.radians(10)
        dq = np.array([np.cos(extra / 2), 0, np.sin(extra / 2), 0])
        desired = np.stack([
            agent_fx._qmul(dq, base[i]) for i in range(n)
        ])
        delta = agent_fx.delta_quat(desired, base)
        for i in range(n):
            got = agent_fx._qmul(delta[i], base[i])
            self.assertAlmostEqual(float(np.dot(got, desired[i])), 1.0, places=7)

    def test_taper_quat_edges_identity(self):
        n = 20
        dq = np.tile(np.array([np.cos(0.1), 0, np.sin(0.1), 0]), (n, 1))
        out = agent_fx.taper_quat_deltas(dq, blend=4)
        self.assertTrue(np.allclose(out[0], [1, 0, 0, 0], atol=1e-9))
        self.assertTrue(np.allclose(out[-1], [1, 0, 0, 0], atol=1e-9))
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=1), 1.0))


class JitterTests(unittest.TestCase):
    def test_zero_phase_preserves_edges(self):
        t = np.linspace(0, 6 * np.pi, 100)
        sig = np.sin(t) + 0.4 * np.random.RandomState(0).randn(100)
        out = agent_fx.clean_jitter_values(sig, strength=0.9, blend=6)
        self.assertAlmostEqual(out[0], sig[0], places=2)
        self.assertAlmostEqual(out[-1], sig[-1], places=2)
        # high-frequency energy reduced inside
        hf = lambda v: np.abs(np.diff(np.diff(v))).mean()
        self.assertLess(hf(out[10:90]), hf(sig[10:90]))


class GroundTests(unittest.TestCase):
    def test_lift_only_pulls_down(self):
        sole = np.array([0.05, 0.03, 0.001, -0.01, 0.04])
        out = agent_fx.ground_height_targets(sole, floor_z=0.0,
                                             rest_clearance=0.0,
                                             mode="lift", blend=0)
        # every floating frame lands exactly on the floor; penetration kept
        self.assertTrue(np.all(out >= -0.011))
        self.assertAlmostEqual(out[2], 0.0, places=6)
        self.assertAlmostEqual(out[3], -0.01, places=6)

    def test_pen_pushes_up(self):
        sole = np.array([0.05, -0.01, -0.02, 0.04])
        out = agent_fx.ground_height_targets(sole, floor_z=0.0, mode="pen",
                                             blend=0)
        self.assertTrue(np.all(out >= -1e-9))

    def test_pelvis_keep_ratio(self):
        n = 8
        pelvis_h = np.full(n, 1.0)
        hip = np.stack([np.zeros(n), np.zeros(n), np.full(n, 0.95)], axis=1)
        ankle = np.stack([np.zeros(n), np.zeros(n), np.full(n, 0.05)], axis=1)
        dz = agent_fx.pelvis_height_corrections(
            pelvis_h, hip, ankle, leg_len=0.8, src_leg_ratio=0.9,
            mode="keep_ratio", blend=0)
        # target dist = 0.72 < current 0.90 → pelvis must sink by ~0.18
        self.assertTrue(np.all(dz < 0))
        self.assertAlmostEqual(dz[3], -0.18, places=2)


class ExemplarTests(unittest.TestCase):
    def test_extract_apply_roundtrip(self):
        n = 12
        base_pos = np.stack([np.zeros(n), np.arange(n) * 0.01,
                             np.full(n, 0.08)], axis=1)
        base_quat = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
        # exemplar: foot lifts +0.02 z for a few frames
        cur_pos = base_pos.copy()
        cur_pos[4:8, 2] += 0.02
        cur_quat = base_quat.copy()
        res = agent_fx.extract_residual(
            base_pos, base_quat, cur_pos, cur_quat, base_pos[0], 0.0)
        # apply onto a different target (same length)
        tgt_pos = np.stack([np.full(n, 1.0), np.arange(n) * 0.02,
                            np.full(n, 0.09)], axis=1)
        out = agent_fx.apply_residual(res, tgt_pos, base_quat, 0.0, blend=2)
        # mid-window the lift should show up (tapered at edges)
        self.assertGreater(out["pos"][5, 2] - tgt_pos[5, 2], 0.01)
        self.assertAlmostEqual(out["pos"][0, 2], tgt_pos[0, 2], places=6)

    def test_dtw_and_match(self):
        a = [0, 0, 5, 5, 0, 0]
        b = [0, 0, 5, 5, 0, 0, 0]
        c = [9, 9, 9, 9, 9, 9]
        self.assertLess(agent_fx.dtw(a, b), agent_fx.dtw(a, c))
        sig = {"duration": 6, "yaw_rate": a}
        cand_same = {"duration": 7, "yaw_rate": b}
        cand_diff = {"duration": 6, "yaw_rate": c}
        self.assertGreater(agent_fx.match_signature(sig, cand_same),
                           agent_fx.match_signature(sig, cand_diff))


class SlerpBridgeTests(unittest.TestCase):
    """hold_pose mode=outlier bridges the bad frames BETWEEN two good neighbours
    (review M5): the neighbours themselves are not part of the bridge."""

    @staticmethod
    def _q(deg):
        a = np.radians(deg)
        return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])

    @staticmethod
    def _deg(q):
        return np.degrees(2 * np.arctan2(q[:, 3], q[:, 0]))

    def test_three_bad_frames_of_a_steady_move(self):
        # 2 deg/frame: good frames at 6 and 14 deg → bad frames are 8/10/12
        out = agent_fx.slerp_series(self._q(6.0), self._q(14.0), 3)
        self.assertTrue(np.allclose(self._deg(out), [8.0, 10.0, 12.0], atol=1e-9),
                        self._deg(out))           # old i/(n-1): [6, 10, 14]

    def test_single_bad_frame_is_the_midpoint(self):
        out = agent_fx.slerp_series(self._q(8.0), self._q(12.0), 1)
        self.assertAlmostEqual(float(self._deg(out)[0]), 10.0, places=9)  # old: 8


class SwingTwistTests(unittest.TestCase):
    """swing_twist_deg with w < 0 (review M28): same rotation, same twist."""

    def test_negative_w_gives_the_same_twist(self):
        a = np.radians(10.0)
        q = np.array([[np.cos(a / 2), 0.0, np.sin(a / 2), 0.0]])     # 10 deg about Y
        pos = agent_fx.swing_twist_deg(q)["twist_deg"][0]
        neg = agent_fx.swing_twist_deg(-q)["twist_deg"][0]           # old: -350
        self.assertAlmostEqual(float(pos), 10.0, places=9)
        self.assertAlmostEqual(float(neg), 10.0, places=9)

    def test_twist_stays_continuous_through_180(self):
        deg = np.arange(170.0, 191.0, 2.0)
        q = np.stack([np.cos(np.radians(deg) / 2), np.zeros_like(deg),
                      np.sin(np.radians(deg) / 2), np.zeros_like(deg)], axis=1)
        tw = agent_fx.swing_twist_deg(q)["twist_deg"]
        wrapped = (tw - deg + 180.0) % 360.0 - 180.0                 # equal mod 360
        self.assertTrue(np.allclose(wrapped, 0.0, atol=1e-9), tw)
        self.assertTrue(np.all(np.abs(tw) <= 180.0 + 1e-9), tw)


if __name__ == "__main__":
    unittest.main()
