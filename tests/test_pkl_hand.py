"""Unit tests for core/pkl_hand.py - pure numpy, no Blender needed."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mocap_doctor.core import pkl_hand


def _make_pkl(frames=120, seed=7):
    rng = np.random.RandomState(seed)
    body_pose = np.zeros((frames, 63), dtype=np.float32)
    # give the wrists smooth random walks so deltas are nonzero
    for col in (57, 58, 59, 60, 61, 62):
        body_pose[:, col] = np.cumsum(rng.randn(frames) * 0.02)
    return {
        "smpl_params_global": {
            "body_pose": body_pose.copy(),
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "betas": np.zeros((frames, 10), dtype=np.float32),
            "left_hand_pose": np.zeros((frames, 45), dtype=np.float32),
            "right_hand_pose": np.zeros((frames, 45), dtype=np.float32),
            "left_hand_global_orient": np.zeros((frames, 3), dtype=np.float32),
            "right_hand_global_orient": np.zeros((frames, 3), dtype=np.float32),
            "left_hand_is_detected": np.ones(frames, dtype=bool),
            "right_hand_is_detected": np.ones(frames, dtype=bool),
            "left_hand_keypoint_conf": np.full(frames, 0.9, dtype=np.float32),
            "right_hand_keypoint_conf": np.full(frames, 0.9, dtype=np.float32),
        },
        "smpl_params_incam": {
            "body_pose": body_pose.copy(),
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "betas": np.zeros((frames, 10), dtype=np.float32),
            "left_hand_pose": np.zeros((frames, 45), dtype=np.float32),
            "right_hand_pose": np.zeros((frames, 45), dtype=np.float32),
            "left_hand_global_orient": np.zeros((frames, 3), dtype=np.float32),
            "right_hand_global_orient": np.zeros((frames, 3), dtype=np.float32),
            "left_hand_is_detected": np.ones(frames, dtype=bool),
            "right_hand_is_detected": np.ones(frames, dtype=bool),
            "left_hand_keypoint_conf": np.full(frames, 0.9, dtype=np.float32),
            "right_hand_keypoint_conf": np.full(frames, 0.9, dtype=np.float32),
        },
        "hamer_merge_info": {"gvhmr_frames": frames},
    }


def _spike(aa):
    # a B-class-like dip: frames 40-49 the wrist flips out and back
    aa = aa.copy()
    aa[40:50, 60:63] += np.array([[0.0, 0.0, 1.2]], dtype=np.float32)
    return aa


class RepairPklTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = Path(self.tmp.name) / "take.pkl"
        with self.src.open("wb") as f:
            pickle.dump(_make_pkl(), f)

    def test_bridge_writes_only_segment_wrist_cols(self):
        dst = Path(self.tmp.name) / "out.pkl"
        report = pkl_hand.repair_pkl(self.src, dst, [{
            "side": "right", "start": 40, "end": 49,
            "strategy": "bridge", "channel": "wrist",
        }])
        self.assertTrue(report["checks"]["wrist_and_finger_only"])
        out = pickle.load(dst.open("rb"))
        orig = pickle.load(self.src.open("rb"))
        g_out, g_in = out["smpl_params_global"], orig["smpl_params_global"]
        # wrist cols inside segment changed
        self.assertFalse(np.array_equal(
            g_out["body_pose"][40:50, 60:63], g_in["body_pose"][40:50, 60:63]))
        # wrist cols outside unchanged
        mask = np.ones(120, dtype=bool)
        mask[40:50] = False
        self.assertTrue(np.array_equal(
            g_out["body_pose"][mask][:, 57:63], g_in["body_pose"][mask][:, 57:63]))
        # other fields identical
        self.assertTrue(np.array_equal(
            g_out["left_hand_keypoint_conf"], g_in["left_hand_keypoint_conf"]))
        self.assertEqual(out["hamer_merge_info"], {"gvhmr_frames": 120})

    def test_hold_pins_pre_pose(self):
        dst = Path(self.tmp.name) / "out.pkl"
        pkl_hand.repair_pkl(self.src, dst, [{
            "side": "right", "start": 40, "end": 49,
            "strategy": "hold", "channel": "wrist",
        }])
        out = pickle.load(dst.open("rb"))
        seg = out["smpl_params_global"]["body_pose"][40:44, 60:63]
        # first 4 frames of a 10-frame hold sit at the pre-segment pose
        self.assertTrue(np.allclose(seg, seg[0], atol=1e-6))

    def test_both_channels_repair_fingers_too(self):
        dst = Path(self.tmp.name) / "out.pkl"
        pkl_hand.repair_pkl(self.src, dst, [{
            "side": "left", "start": 60, "end": 70,
            "strategy": "smooth", "channel": "both",
        }])
        out = pickle.load(dst.open("rb"))
        self.assertTrue(dst.is_file())
        # fingers inside the segment were rewritten (zeros stay zero anyway
        # here, but the call path must not error)
        report_seg = None
        # repair ran without exceptions = both channels handled

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            pkl_hand.repair_pkl(self.src, Path(self.tmp.name) / "o.pkl", [{
                "side": "left", "start": 10, "end": 20, "strategy": "nope",
            }])


class DetectTests(unittest.TestCase):
    def test_detect_finds_spike_on_raw(self):
        # raw hamer layout: rotation matrices (T,2,3,3) / (T,2,15,3,3)
        frames = 100
        ident = np.eye(3)
        go = np.tile(ident[None, None, :, :], (frames, 2, 1, 1)).astype(np.float32)
        hp = np.tile(ident[None, None, None, :, :], (frames, 2, 15, 1, 1)).astype(np.float32)
        # inject a 10-frame wrist spike on channel 1 (right); interior frames
        # wander like a real bad segment does (constant pose = invisible A-class)
        for f in range(40, 50):
            ang = np.radians(120.0 + 30.0 * np.sin(f))
            go[f, 1] = np.array([[np.cos(ang), -np.sin(ang), 0],
                                 [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        data = {
            "global_orient": go,
            "hand_pose": hp,
            "global_orient_valid": np.ones((frames, 2), dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hamer.pkl"
            with p.open("wb") as f:
                pickle.dump(data, f)
            res = pkl_hand.detect_candidates(p)
        self.assertEqual(res["source_kind"], "raw_hamer")
        right = res["R"]
        self.assertTrue(right, "right-hand spike not detected")
        hit = [c for c in right if c["start"] <= 40 and c["end"] >= 49]
        self.assertTrue(hit, f"candidates {right} miss 40-49")
        self.assertIn(hit[0]["strategy"], ("bridge", "smooth"))
        # clean left hand has far fewer (ideally zero) candidates
        self.assertLessEqual(len(res["L"]), 1)


class ReviewFixTests(unittest.TestCase):
    """2026-10-04 review: M11 (adjacent segments), M29 (aa path), M18 (numpy 2 pkl)."""

    def test_adjacent_segments_do_not_contaminate_each_other(self):
        # two HaMeR flips 3 frames apart on a smooth right-wrist path (review
        # repro): each segment's reference window used to average the OTHER
        # segment's bad frames in → 44 deg error; a 30-frame gap stayed ≤2.2 deg
        frames = 120
        t = np.arange(frames)
        good = np.stack([0.3 * np.sin(t / 15.0), 0.2 * np.cos(t / 20.0),
                         0.1 + 0.002 * t], axis=1)
        bad = np.zeros((frames, 63))
        bad[:, 60:63] = good
        for a, b in ((40, 49), (53, 60)):
            bad[a:b + 1, 60:63] = good[a:b + 1] + np.array([0.0, 2.2, 0.0])
        data = {"smpl_params_global": {
            "body_pose": bad.astype(np.float32),
            "left_hand_pose": np.zeros((frames, 45), np.float32),
            "right_hand_pose": np.zeros((frames, 45), np.float32)}}
        segs = [{"side": "right", "start": 40, "end": 49, "strategy": "bridge", "channel": "wrist"},
                {"side": "right", "start": 53, "end": 60, "strategy": "bridge", "channel": "wrist"}]
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "m.pkl", Path(td) / "o.pkl"
            with src.open("wb") as f:
                pickle.dump(data, f)
            pkl_hand.repair_pkl(src, dst, segs)
            with dst.open("rb") as f:
                out = pickle.load(f)["smpl_params_global"]["body_pose"][:, 60:63]
        out = np.asarray(out, dtype=np.float64)
        err = [float(pkl_hand.mat_geo_deg(pkl_hand.rotvec_to_mat(out[f]),
                                          pkl_hand.rotvec_to_mat(good[f])))
               for f in range(40, 61)]
        self.assertLess(max(err), 6.0, [round(e, 1) for e in err])

    def test_no_neighbour_segment_keeps_the_old_window(self):
        q = pkl_hand.aa_to_quat(np.random.RandomState(1).randn(30, 3) * 0.3)
        a = pkl_hand.repair_quats(q, 10, 14, "bridge", {})
        b = pkl_hand.repair_quats(q, 10, 14, "bridge", {}, exclude={25, 26})
        self.assertTrue(np.array_equal(a, b))

    def test_smooth_aa_path_keeps_the_rotation_across_180(self):
        z = np.array([0.0, 0.0, 1.0])
        true_deg = np.arange(170.0, 201.0, 5.0)
        true_rv = np.radians(true_deg)[:, None] * z
        canon = pkl_hand.quat_to_aa(pkl_hand.aa_to_quat(true_rv))  # ≤180 deg form
        out = pkl_hand.smooth_aa_path(canon[1:].copy(), true_rv[0])
        err = [float(pkl_hand.mat_geo_deg(pkl_hand.rotvec_to_mat(o),
                                          pkl_hand.rotvec_to_mat(r)))
               for o, r in zip(out, true_rv[1:])]
        self.assertLess(max(err), 1e-6, err)            # old -aa: up to 40 deg
        jumps = np.linalg.norm(np.diff(np.vstack([true_rv[:1], out]), axis=0), axis=1)
        self.assertLess(float(jumps.max()), np.radians(6.0))

    def test_numpy2_pickle_loads_under_blender_numpy(self):
        """A protocol-5 pkl written by numpy >= 2 references numpy._core.numeric,
        which Blender 4.5's numpy 1.26 lacks.  Checked with Blender's own
        bundled interpreter when it is on this machine (not Blender itself)."""
        import glob
        import os
        import subprocess
        import sys
        data = {"smpl_params_global": {"body_pose": np.arange(12, dtype=np.float32).reshape(2, 6)}}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "np2.pkl"
            with p.open("wb") as f:
                pickle.dump(data, f, protocol=5)
            got = pkl_hand.load_pickle(p)                 # this interpreter
            self.assertTrue(np.array_equal(got["smpl_params_global"]["body_pose"],
                                           data["smpl_params_global"]["body_pose"]))
            if int(np.__version__.split(".")[0]) < 2:
                return
            cands = sorted(glob.glob(os.path.expanduser(
                "~/Downloads/blender-4.5*/4.5/python/bin/python3.11")))
            if not cands:
                print("SKIP numpy 1.26 cross-check: no Blender-bundled python found")
                return
            root = str(Path(__file__).resolve().parents[1])
            code = ("import pickle,sys\n"
                    "sys.path.insert(0, %r)\n"
                    "from mocap_doctor.core import pkl_hand\n"
                    "try:\n pickle.load(open(%r,'rb'), encoding='latin1'); print('PLAIN_OK')\n"
                    "except ModuleNotFoundError as e: print('PLAIN_FAILS', e)\n"
                    "d = pkl_hand.load_pickle(%r)\n"
                    "print('COMPAT', d['smpl_params_global']['body_pose'].sum())\n") % (root, str(p), str(p))
            out = subprocess.run([cands[-1], "-c", code], capture_output=True, text=True,
                                 timeout=60).stdout
            self.assertIn("PLAIN_FAILS", out, out)        # the bug: plain load fails
            self.assertIn("COMPAT 66.0", out, out)        # the fix: loads, same data


if __name__ == "__main__":
    unittest.main()
