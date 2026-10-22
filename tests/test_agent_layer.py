"""Unit tests for the agent input/output layer - pure numpy, no Blender."""
import unittest

import numpy as np

from mocap_doctor.core import accent, agent_query, agent_signals


def _fake_bake(frames=60):
    n = frames
    t = np.arange(n, dtype=np.float64)
    pos = {
        "hips": np.stack([t * 0, t * 0, np.full(n, 0.95)], axis=1),
        "left_foot": np.stack([np.zeros(n), np.zeros(n), np.full(n, 0.08)], axis=1),
        "right_foot": np.stack([np.zeros(n), np.zeros(n), np.full(n, 0.08)], axis=1),
    }
    # left foot steps forward at frame 30 then plants
    pos["left_foot"][:, 1] = np.where(t < 30, t * 0.02, 0.6)
    points = {
        "foot.L.heel": pos["left_foot"] - np.array([0, 0.05, 0.02]),
        "foot.L.ball": pos["left_foot"] + np.array([0, 0.10, -0.02]),
        "foot.L.toe": pos["left_foot"] + np.array([0, 0.20, -0.04]),
        "foot.R.heel": pos["right_foot"] - np.array([0, 0.05, 0.02]),
        "foot.R.ball": pos["right_foot"] + np.array([0, 0.10, -0.02]),
        "foot.R.toe": pos["right_foot"] + np.array([0, 0.20, -0.04]),
    }
    quat = {"hips": np.tile(np.array([1.0, 0, 0, 0]), (n, 1)),
            "left_foot": np.tile(np.array([1.0, 0, 0, 0]), (n, 1))}
    return {"frames": np.arange(1, n + 1), "pos": pos, "point_pos": points,
            "quat": quat, "frame_start": 1, "frame_end": n}


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.bake = _fake_bake()
        self.signals = agent_signals.compute_signals(
            self.bake, floor_z=0.0,
            intervals={"contact.L": [[30, 55]], "air": [[10, 14]]},
        )

    def test_foot_channels_present(self):
        for name in ("foot.L.toe_h", "foot.L.yaw", "foot.L.yaw_rate",
                     "foot.L.pivot_idx", "foot.L.pen", "pelvis.jerk"):
            self.assertIn(name, self.signals, name)

    def test_masks(self):
        c = self.signals["contact.L"]["values"]
        self.assertEqual(c.sum(), 26.0)   # frames 30..55 inclusive
        air = self.signals["air"]["values"]
        self.assertEqual(air.sum(), 5.0)

    def test_masks_accept_dict_items(self):
        # regression: scene_intervals produces {"id","start","end"} dicts
        signals = agent_signals.compute_signals(
            self.bake, floor_z=0.0,
            intervals={"contact.L": [{"id": 0, "start": 30, "end": 40}]},
        )
        self.assertEqual(signals["contact.L"]["values"].sum(), 11.0)

    def test_pivot_stationary_while_moving(self):
        # planted at 30-55 the foot doesn't move → some point stays still
        still = self.signals["foot.L.pivot_still"]["values"]
        self.assertGreater(still[35:50].mean(), 0.8)
        # during the step (frames <30) nothing is pinned
        self.assertLess(still[5:25].mean(), 0.5)


class QueryTests(unittest.TestCase):
    def setUp(self):
        bake = _fake_bake()
        signals = agent_signals.compute_signals(
            bake, floor_z=0.0,
            intervals={"contact.L": [[30, 55]]},
        )
        self.store = agent_query.build_store(
            signals, bake["frames"],
            intervals={"contact.L": [{"id": 0, "start": 30, "end": 55,
                                      "tag": "planted"}]},
            positions=bake["pos"], floor_z=0.0,
        )

    def test_envelope_shape(self):
        out = agent_query.get_overview(self.store)
        for key in ("summary", "data", "warnings", "truncated", "hint"):
            self.assertIn(key, out)

    def test_overview(self):
        ov = agent_query.get_overview(self.store)["data"]
        self.assertEqual(ov["frames"], [1, 60])
        self.assertTrue(any(s["name"] == "pelvis.h" for s in ov["channels"]))
        self.assertEqual(ov["intervals"]["contact.L"]["count"], 1)

    def test_describe_interval_has_flags(self):
        card = agent_query.describe(self.store, "contact.L:0")
        self.assertIn("flags", card["data"])
        self.assertEqual(card["data"]["frames"], [30, 55])
        self.assertTrue(card["data"]["stats"])

    def test_get_series_downsamples(self):
        out = agent_query.get_series(
            self.store, ["foot.L.speed_xy"], [1, 60], max_points=20)
        data = out["data"]
        self.assertLessEqual(len(data["frames"]), 20)
        self.assertIn("foot.L.speed_xy", data["series"])
        self.assertTrue(out["truncated"])

    def test_find_events_declarative(self):
        res = agent_query.find_events(self.store, {
            "all": [{"ch": "contact.L", "op": "==", "v": 1}],
            "min_len": 2,
        })["data"]
        self.assertEqual(res["matches"][0]["frames"], [30, 55])

    def test_compare_two_ranges(self):
        out = agent_query.compare(
            self.store, "foot.L.speed_xy", [10, 20], [35, 50])["data"]
        self.assertIn("peak_ratio", out)

    def test_bad_channel_suggests(self):
        with self.assertRaises(agent_query.AgentQueryError) as ctx:
            agent_query.get_series(self.store, ["foot.L.yaw_rat"], [1, 10])
        self.assertEqual(ctx.exception.code, "E_UNKNOWN")
        self.assertIn("yaw_rate", ctx.exception.fix)

    def test_snapshot(self):
        snap = agent_query.snapshot(self.store, 30)["data"]
        self.assertIn("hips", snap["positions"])

    def test_range_error_code(self):
        with self.assertRaises(agent_query.AgentQueryError) as ctx:
            agent_query.describe(self.store, [900, 1000])
        self.assertEqual(ctx.exception.code, "E_RANGE")


class AccentTests(unittest.TestCase):
    def setUp(self):
        n = 60
        self.t = np.arange(n, dtype=np.float64)
        # raw has a sharp snap at frame 30; cur is over-smoothed
        self.raw = np.where(self.t < 30, self.t * 0.0,
                            np.minimum(1.0, (self.t - 30) * 0.5))
        kernel = np.ones(7) / 7
        self.cur = np.convolve(np.pad(self.raw, 3, mode="edge"),
                               kernel, mode="valid")[:n]

    def test_hf_reinject_restores_sharpness(self):
        before = accent.accent_metrics(self.cur, self.cur, self.raw)
        out = accent.hf_reinject(self.cur, self.raw, strength=0.8)
        after = accent.accent_metrics(self.cur, out, self.raw)
        self.assertGreater(after["peak_acc_after"], before["peak_acc_after"])
        # endpoints pinned
        self.assertAlmostEqual(out[0], self.cur[0], places=9)
        self.assertAlmostEqual(out[-1], self.cur[-1], places=9)

    def test_ease_reshape_endpoints(self):
        out = accent.ease_reshape(self.cur, 30, pre=10, post=10,
                                 strength=0.6, overshoot=0.2)
        self.assertAlmostEqual(out[0], self.cur[0], places=9)
        self.assertAlmostEqual(out[-1], self.cur[-1], places=9)
        # approach should be steeper right before impact
        self.assertGreater(abs(out[30] - out[29]),
                           abs(self.cur[30] - self.cur[29]))

    def test_retime_endpoints_and_shape(self):
        out = accent.retime(self.cur, attack_speed=1.8, split=0.4)
        self.assertAlmostEqual(out[0], self.cur[0], places=9)
        self.assertAlmostEqual(out[-1], self.cur[-1], places=9)

    def test_retime_rejects_bad_ratio(self):
        with self.assertRaises(ValueError):
            accent.retime(self.cur, attack_speed=4.0, split=0.5)

    def test_refilter_moves_toward_mild_raw(self):
        out = accent.refilter(self.cur, self.raw, strength=0.9)
        mid = slice(25, 35)
        self.assertGreater(
            float(np.abs(out[mid] - self.raw[mid]).mean() <
                  np.abs(self.cur[mid] - self.raw[mid]).mean()),
            0.0,
        )

    def test_quat_reinject(self):
        # identity vs slowly-rotating raw on one axis
        n = 30
        cur = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
        half = np.radians(5 * np.sin(np.linspace(0, np.pi, n)))
        raw = np.stack([np.cos(half), np.zeros(n), np.sin(half),
                        np.zeros(n)], axis=1)
        out = accent.hf_reinject_quats(cur, raw, strength=0.8, blend=2)
        norms = np.linalg.norm(out, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-9))
        # sign continuity maintained
        self.assertTrue(all(np.dot(out[i - 1], out[i]) >= 0 for i in range(1, n)))


class JointAngleTests(unittest.TestCase):
    def test_swing_twist_pure_flex(self):
        from mocap_doctor.core import agent_fx
        # 30° rotation about X = pure flexion
        a = np.radians(30)
        q = np.array([[np.cos(a / 2), np.sin(a / 2), 0, 0]])
        ja = agent_fx.swing_twist_deg(q)
        self.assertAlmostEqual(ja["swing_deg"][0], 30.0, places=4)
        self.assertAlmostEqual(abs(ja["flex_deg"][0]), 30.0, places=4)
        self.assertAlmostEqual(abs(ja["twist_deg"][0]), 0.0, places=4)

    def test_swing_twist_pure_twist(self):
        from mocap_doctor.core import agent_fx
        a = np.radians(45)
        q = np.array([[np.cos(a / 2), 0, np.sin(a / 2), 0]])  # about bone-Y
        ja = agent_fx.swing_twist_deg(q)
        self.assertAlmostEqual(abs(ja["twist_deg"][0]), 45.0, places=4)
        self.assertAlmostEqual(ja["swing_deg"][0], 0.0, places=3)


class TimelineMarkerTests(unittest.TestCase):
    """list_timeline_markers: the user's M-key markers as anchor frames."""

    INTERVALS = {
        "contact.L": [{"id": 0, "start": 480, "end": 520}],
        "air": [{"id": 0, "start": 614, "end": 628}],
        "jitter.L": [{"id": 0, "start": 490, "end": 560}],
    }
    MARKERS = [
        {"name": "落地", "frame": 620},
        {"name": "出拳", "frame": 505},
        {"name": "", "frame": 100},
    ]

    def test_sorted_and_joined_to_intervals(self):
        env = agent_query.list_timeline_markers(
            self.MARKERS, self.INTERVALS, with_intervals=True)
        items = env["data"]["items"]
        self.assertEqual([i["frame"] for i in items], [100, 505, 620])
        self.assertEqual(items[0]["covered_by"], [])
        # 505 sits inside both the left-hand range and the planted span
        kinds = {c["kind"] for c in items[1]["covered_by"]}
        self.assertEqual(kinds, {"contact.L", "jitter.L"})
        self.assertEqual(items[2]["covered_by"], [{"kind": "air", "start": 614, "end": 628}])
        self.assertIn("505 出拳", env["summary"])

    def test_frame_range_and_name_filters(self):
        env = agent_query.list_timeline_markers(
            self.MARKERS, self.INTERVALS, frame_range=[500, 510])
        self.assertEqual([i["frame"] for i in env["data"]["items"]], [505])
        env = agent_query.list_timeline_markers(self.MARKERS, None, name="出拳")
        self.assertEqual([i["frame"] for i in env["data"]["items"]], [505])
        # filtered-to-nothing says so instead of looking like "no markers"
        env = agent_query.list_timeline_markers(self.MARKERS, None, name="没有这个")
        self.assertEqual(env["data"]["total"], 0)
        self.assertIn("过滤", env["hint"])
        self.assertTrue(env["warnings"])

    def test_no_markers_hint_points_at_the_next_step(self):
        env = agent_query.list_timeline_markers([], None)
        self.assertEqual(env["data"]["total"], 0)
        self.assertIn("analyze_motion", env["hint"])
        self.assertFalse(env["warnings"])

    def test_with_intervals_off(self):
        env = agent_query.list_timeline_markers(
            self.MARKERS, self.INTERVALS, with_intervals=False)
        self.assertNotIn("covered_by", env["data"]["items"][0])

    def test_bad_frame_range_is_E_RANGE(self):
        for bad in ([5], "505-560", [560, 505]):
            with self.assertRaises(agent_query.AgentQueryError) as ctx:
                agent_query.list_timeline_markers(self.MARKERS, None, frame_range=bad)
            self.assertEqual(ctx.exception.code, "E_RANGE")


if __name__ == "__main__":
    unittest.main()
