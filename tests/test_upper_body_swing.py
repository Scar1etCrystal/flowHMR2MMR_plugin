"""上半身对齐的纯数学部分（core.target.upper_body_swing），不需要 Blender。

坐标：Z 向上，Teto 与源都面朝 -Y，左髋在 +X（与 Blender 里 .L 在 +X 一致）；
lateral = 右髋 - 左髋 的水平方向（= -X），forward = up × lateral（= -Y）。
"""

import math
import unittest

import numpy as np

from mocap_doctor.core import target


def frames(count, point):
    return np.repeat(np.asarray(point, dtype=float)[None], count, axis=0)


def scene(count=30, src_shoulder=(0.0, 0.0, 1.5), teto_shoulder=(0.0, 0.0, 1.3)):
    """Teto: hips-mid at z=0.95, pivot (hips head) at z=1.03; source hips-mid at z=0.9."""
    return dict(
        pivot=frames(count, (0.0, 0.0, 1.03)),
        shoulders=frames(count, teto_shoulder),
        hip_l=frames(count, (0.08, 0.0, 0.95)),
        hip_r=frames(count, (-0.08, 0.0, 0.95)),
        src_shoulders=frames(count, src_shoulder),
        src_hip_l=frames(count, (0.09, 0.0, 0.9)),
        src_hip_r=frames(count, (-0.09, 0.0, 0.9)),
    )


def lateral_after(data, result):
    arm = data["shoulders"] - data["pivot"]
    new = data["pivot"] + target._rotate_rows(result["rotvec"], arm)
    hip = (data["hip_l"] + data["hip_r"]) / 2.0
    return (new - hip) @ np.array([-1.0, 0.0, 0.0]), (new - hip) @ np.array([0.0, -1.0, 0.0])


class UpperBodySwingTests(unittest.TestCase):
    def test_matching_offsets_need_no_swing(self):
        data = scene()
        result = target.upper_body_swing(**data, torso_ratio=1.0, smooth_sigma=0)
        self.assertLess(np.abs(result["rotvec"]).max(), 1e-9)
        self.assertEqual(result["clamped_frames"], 0)

    def test_lateral_offset_is_copied_one_to_one(self):
        """Source shoulders 4 cm to the subject's right of its hips: Teto's land 4 cm right too."""
        data = scene(src_shoulder=(-0.04, 0.0, 1.5))
        result = target.upper_body_swing(**data, smooth_sigma=0)
        lat, fwd = lateral_after(data, result)
        np.testing.assert_allclose(lat, 0.04, atol=1e-9)
        np.testing.assert_allclose(fwd, 0.0, atol=1e-9)
        # the swing is a roll about the forward (-Y) axis through the hips head
        axis = result["rotvec"][0] / np.linalg.norm(result["rotvec"][0])
        self.assertAlmostEqual(abs(axis[1]), 1.0, places=9)
        self.assertAlmostEqual(math.degrees(np.linalg.norm(result["rotvec"][0])), math.degrees(math.asin(0.04 / 0.27)), places=6)
        self.assertAlmostEqual(result["shoulder_lateral_err_cm"]["before"]["p50"], 4.0, places=6)
        self.assertAlmostEqual(result["shoulder_lateral_err_cm"]["after"]["p95"], 0.0, places=6)

    def test_forward_offset_is_scaled_by_the_torso_ratio(self):
        data = scene(src_shoulder=(0.0, -0.10, 1.5))
        result = target.upper_body_swing(**data, torso_ratio=0.5, smooth_sigma=0)
        lat, fwd = lateral_after(data, result)
        np.testing.assert_allclose(fwd, 0.05, atol=1e-9)
        np.testing.assert_allclose(lat, 0.0, atol=1e-9)

    def test_measured_torso_ratio(self):
        data = scene()
        result = target.upper_body_swing(**data, smooth_sigma=0)
        self.assertAlmostEqual(result["torso_ratio"], round(0.35 / 0.6, 4), places=4)

    def test_fuse_clamps_the_swing(self):
        data = scene(src_shoulder=(-0.30, 0.0, 1.45))
        result = target.upper_body_swing(**data, smooth_sigma=0, max_swing_deg=10.0)
        self.assertAlmostEqual(math.degrees(np.linalg.norm(result["rotvec"], axis=1).max()), 10.0, places=6)
        self.assertEqual(result["clamped_frames"], 30)

    def test_lying_source_fades_out(self):
        """Floor work: the source torso is 80 deg off vertical, no correction."""
        lean = math.radians(80.0)
        data = scene(src_shoulder=(-0.6 * math.sin(lean), 0.0, 0.9 + 0.6 * math.cos(lean)))
        result = target.upper_body_swing(**data, smooth_sigma=0)
        self.assertLess(np.abs(result["rotvec"]).max(), 1e-12)
        self.assertEqual(result["faded_frames"], 30)

    def test_vertical_hip_line_gets_no_correction(self):
        data = scene(src_shoulder=(-0.04, 0.0, 1.5))
        data["src_hip_r"] = data["src_hip_l"] + np.array([0.0, 0.0, -0.18])
        result = target.upper_body_swing(**data, smooth_sigma=0)
        self.assertLess(np.abs(result["rotvec"]).max(), 1e-12)
        self.assertEqual(result["degenerate_frames"], 30)

    def test_smoothing_spreads_a_one_frame_spike(self):
        data = scene(count=21)
        data["src_shoulders"][10] = (-0.04, 0.0, 1.5)
        raw = target.upper_body_swing(**data, torso_ratio=1.0, smooth_sigma=0)["rotvec"]
        smooth = target.upper_body_swing(**data, torso_ratio=1.0, smooth_sigma=1.5)["rotvec"]
        self.assertLess(np.linalg.norm(smooth[10]), 0.5 * np.linalg.norm(raw[10]))
        self.assertGreater(np.linalg.norm(smooth[9]), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(smooth.sum(axis=0))), float(np.linalg.norm(raw.sum(axis=0))), places=9)


if __name__ == "__main__":
    unittest.main()
