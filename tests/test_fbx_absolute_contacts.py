"""Route (B): absolute joints from final.fbx → official step-5 velocity term.

Standalone (numpy only, no bpy) so it runs with plain python:

    python tests/test_fbx_absolute_contacts.py

It pins the four measured facts the wiring depends on, then exercises the guard
in both directions.  Skips (exit 0) when the artifacts it needs are absent, so a
checkout without the FlowHMR takes does not turn into a failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from mocap_doctor.core import foot_contacts as FC  # noqa: E402

PROBE = os.path.join(REPO, ".fbx_axis_probe", "fbx_joints.npz")
OFFICIAL = r"F:\flowhmr_official_out\t999_OFFICIAL_post.npz"
OTHER_TAKE = r"F:\flowhmr_official_out\video_HaMeRhands_leftfixed.npz"
OTHER_FBX = r"F:\flowhmr_official_out\video_HaMeRhands_leftfixed.fbx"

# t999_H4 fbx (1000 frames, bone heads, Blender world) vs the official 995-frame
# post-processed take.
MEASURED = {
    "pelvis_height_mean": 0.8973,      # criterion 1 (npz says 0.8921)
    "lowest_foot_joint": 0.0153,       # criterion 2 (npz says -0.0126)
    "offset_scan_mean_cm": {0: 4.749, 1: 5.290, 2: 6.320, 5: 9.752, 8: 12.710},
    "best_offset": 0,
}

checks: list[bool] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    checks.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + " :: " + str(detail), flush=True)


def _skip(reason: str) -> int:
    print(f"SKIP :: {reason}")
    return 0


def part_axis(probe: dict) -> None:
    """Criteria 1 and 2, plus the sign-combination ranking."""
    pos = np.asarray(probe["positions"], dtype=np.float64)      # raw blender world
    names = [str(n) for n in probe["names"]]
    idx = {n: i for i, n in enumerate(names)}
    npz = FC.blender_world_to_npz(pos)
    pelvis = npz[:, idx["Pelvis"], 1]
    feet = [idx[n] for n in FC.OFFICIAL_CONTACT_JOINTS]
    lowest = npz[:, feet, 1].min()
    check("criterion 1: pelvis height mean ~0.892 m", abs(pelvis.mean() - 0.892) < 0.01,
          f"measured {pelvis.mean():.4f} m (target 0.8921)")
    check("criterion 2: lowest foot joint ~0 m", abs(lowest) < 0.03,
          f"measured {lowest:+.4f} m (npz reference -0.0126)")
    check("measured values still reproduced", 
          abs(pelvis.mean() - MEASURED["pelvis_height_mean"]) < 0.002
          and abs(lowest - MEASURED["lowest_foot_joint"]) < 0.002,
          f"pelvis {pelvis.mean():.4f} vs {MEASURED['pelvis_height_mean']}, "
          f"foot {lowest:+.4f} vs {MEASURED['lowest_foot_joint']:+.4f}")

    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    ranking = {}
    ref = np.load(OFFICIAL, allow_pickle=True)
    local = np.asarray(ref["local_joints_positions"], dtype=np.float64)
    JOINT_NAMES = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
                   'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
                   'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
                   'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
    JOINT_NAMES += [f"L_{a}{i}" for a in ("Index", "Middle", "Pinky", "Ring", "Thumb")
                    for i in (1, 2, 3)]
    JOINT_NAMES += [f"R_{a}{i}" for a in ("Index", "Middle", "Pinky", "Ring", "Thumb")
                    for i in (1, 2, 3)]
    shared = [n for n in JOINT_NAMES if n in idx]
    fc = [idx[n] for n in shared]
    nc = [JOINT_NAMES.index(n) for n in shared]
    rel_npz = local[:, nc, :] - local[:, [0], :]
    for label, arr in {
        "(+X,+Z,+Y)": np.stack([x, z, y], axis=-1),
        "(+X,+Z,-Y)": np.stack([x, z, -y], axis=-1),
        "(-X,+Z,-Y)": np.stack([-x, z, -y], axis=-1),
        "(-X,+Z,+Y)": np.stack([-x, z, y], axis=-1),
    }.items():
        rel = arr[:, fc, :] - arr[:, [idx["Pelvis"]], :]
        n = min(local.shape[0], pos.shape[0])
        d = np.linalg.norm(rel[:n] - rel_npz[:n], axis=-1)
        ranking[label] = 100 * float(d.mean())
    winner = min(ranking, key=ranking.get)
    check("axis convention (-X,+Z,+Y) wins the sign scan", winner == "(-X,+Z,+Y)",
          "  ".join(f"{k}={v:.1f}cm" for k, v in sorted(ranking.items(), key=lambda kv: kv[1])))

    # frame correspondence scan
    rel_fbx = np.stack([-x, z, y], axis=-1)
    rel_fbx = rel_fbx[:, fc, :] - rel_fbx[:, [idx["Pelvis"]], :]
    scan = {}
    for offv in range(0, 9):
        n = min(local.shape[0], pos.shape[0] - offv)
        d = np.linalg.norm(rel_fbx[offv:offv + n] - rel_npz[:n], axis=-1)
        scan[offv] = 100 * float(d.mean())
    best = min(scan, key=scan.get)
    check("frame offset minimum is 0 (nothing cropped at the start)", best == 0,
          "  ".join(f"off{k}={v:.3f}cm" for k, v in sorted(scan.items())))
    check("scan is monotonic after the minimum (5 extra frames at the end)",
          all(scan[i] <= scan[i + 1] + 0.05 for i in range(0, 8)),
          f"off0={scan[0]:.3f} off8={scan[8]:.3f}")
    check("offset-0 error is centimetre level (< 6 cm mean)", scan[0] < 6.0, f"{scan[0]:.3f} cm")
    check("recorded scan values still hold",
          all(abs(scan[k] - v) < 0.05 for k, v in MEASURED["offset_scan_mean_cm"].items()),
          {k: round(scan[k], 3) for k in MEASURED["offset_scan_mean_cm"]})


def part_guard(probe_path: str) -> None:
    """Positive pairing, cross-take refusal, and the tolerance boundary."""
    probe = dict(np.load(probe_path, allow_pickle=True))
    tmpdir = os.path.join(REPO, ".fbx_axis_probe")

    # --- positive: the real pairing ---------------------------------------
    report = FC.official_planted_ranges(OFFICIAL, frame_offset=1, joints_path=probe_path)
    check("(B) is used for the velocity term",
          report["velocity_source"] == "fbx_absolute" and not report["velocity_is_guess"],
          report["velocity_source"])
    guard = report["take_guard"]
    check("same-take guard accepts 1000 vs 995", guard["delta_frames"] == 5
          and guard["tolerance_frames"] == 8 and guard["take"] == "t999",
          f"take={guard['take']} delta={guard['delta_frames']} tol={guard['tolerance_frames']}")
    check("planted intervals produced", 
          report["planted_frames"]["L"] > 0 and report["planted_frames"]["R"] > 0,
          f"L {report['planted_frames']['L']} 帧/{report['segment_counts']['L']} 段, "
          f"R {report['planted_frames']['R']} 帧/{report['segment_counts']['R']} 段, "
          f"air {report['air_segments']} 段")
    print("      velocity_note:", report["velocity_note"])

    # air must be exactly the complement of (L planted or R planted)
    # frame_offset=1 means npz index i == Blender frame i+1, so the window is
    # [offset, offset + T - 1].
    lo = int(report["frame_offset"])
    hi = lo + report["frames_in_npz"] - 1
    planted_frames = set()
    for side in ("L", "R"):
        for a, b in report["planted"][side]:
            planted_frames.update(range(a, b + 1))
    air_frames = set()
    for a, b in report["air"]:
        air_frames.update(range(a, b + 1))
    window = set(range(lo, hi + 1))
    check("ground_feet complement: air == window - (L or R)",
          air_frames == window - planted_frames,
          f"air {len(air_frames)} frames, complement {len(window - planted_frames)}, "
          f"window {len(window)}")

    # --- negative 1: another take ----------------------------------------
    other = dict(probe)
    other[FC.JOINTS_SOURCE_KEY] = OTHER_FBX
    p_other = os.path.join(tmpdir, "TMP_other_take_fbx_joints.npz")
    np.savez(p_other, **other)
    try:
        FC.official_planted_ranges(OFFICIAL, joints_path=p_other)
        check("cross-take pairing is refused", False, "no exception raised")
    except FC.TakeMismatchError as exc:
        check("cross-take pairing is refused", True, "")
        print("      negative-case message:", str(exc))
    finally:
        os.remove(p_other)

    # --- negative 1b: real other-take joints (3794 frames) ----------------
    if os.path.isfile(OTHER_TAKE):
        ref = np.load(OTHER_TAKE, allow_pickle=True)
        if "local_joints_positions" in ref:
            p_real = os.path.join(tmpdir, "TMP_real_other_take_fbx_joints.npz")
            np.savez(p_real, positions=np.asarray(ref["local_joints_positions"]),
                     names=np.array(probe[FC.JOINTS_NAMES_KEY], dtype=object),
                     frames=np.arange(1, np.asarray(ref["local_joints_positions"]).shape[0] + 1),
                     source_fbx=OTHER_FBX)
            try:
                FC.official_planted_ranges(OFFICIAL, joints_path=p_real)
                check("cross-take (real 3794-frame take) is refused", False, "no exception")
            except FC.TakeMismatchError as exc:
                check("cross-take (real 3794-frame take) is refused", True, "")
                print("      negative-case message:", str(exc))
            finally:
                os.remove(p_real)

    # --- negative 2: tolerance boundary 8 ok / 9 refused ------------------
    base_pos = np.asarray(probe[FC.JOINTS_POSITIONS_KEY])
    frames = np.asarray(probe["frames"]) if "frames" in probe else np.arange(1, base_pos.shape[0] + 1)
    for n_rows, should_pass in ((987, True), (986, False)):
        p = os.path.join(tmpdir, f"TMP_short{n_rows}_fbx_joints.npz")
        np.savez(p, positions=base_pos[:n_rows], names=probe[FC.JOINTS_NAMES_KEY],
                 frames=frames[:n_rows], source_fbx=probe[FC.JOINTS_SOURCE_KEY])
        try:
            FC.official_planted_ranges(OFFICIAL, joints_path=p)
            ok, msg = should_pass, "accepted"
        except FC.TakeMismatchError as exc:
            ok, msg = not should_pass, str(exc)
        finally:
            os.remove(p)
        check(f"tolerance boundary: {n_rows} rows ({995 - n_rows} frame delta) "
              f"{'accepted' if should_pass else 'refused'}", ok, msg[:160])

    # --- fallback: no joints file -> the npz approximation, labelled ------
    fb = FC.official_planted_ranges(OFFICIAL, frame_offset=1)
    check("fallback keeps working and is labelled a guess",
          fb["velocity_is_guess"] and "在猜" in fb["velocity_note"],
          fb["velocity_source"])
    same = (fb["planted"]["L"] == report["planted"]["L"]
            and fb["planted"]["R"] == report["planted"]["R"])
    print("      (B) vs fallback planted ranges identical:", same,
          f"| L (B)={len(report['planted']['L'])} 段 fallback={len(fb['planted']['L'])} 段",
          f"| R (B)={len(report['planted']['R'])} 段 fallback={len(fb['planted']['R'])} 段")

    def frames_of(rep: dict, side: str) -> set:
        out = set()
        for a, b in rep["planted"][side]:
            out.update(range(int(a), int(b) + 1))
        return out

    for side in ("L", "R"):
        b_f, f_f = frames_of(report, side), frames_of(fb, side)
        print(f"      planted {side}: (B) {len(b_f)} 帧 / fallback {len(f_f)} 帧 / "
              f"交集 {len(b_f & f_f)} / 差异 {len(b_f ^ f_f)} 帧")

    # --- negative 3: two real takes in the same folder (t999 vs t1499) ----
    other_official = OFFICIAL.replace("t999_", "t1499_")
    if os.path.isfile(other_official):
        ref = np.load(other_official, allow_pickle=True)
        if "local_joints_positions" in ref:
            pos_other = np.asarray(ref["local_joints_positions"])
            p = os.path.join(tmpdir, "TMP_t1499_fbx_joints.npz")
            np.savez(p, positions=pos_other,
                     names=np.array(probe[FC.JOINTS_NAMES_KEY], dtype=object),
                     frames=np.arange(1, pos_other.shape[0] + 1),
                     source_fbx=r"F:\flowhmr_pack\outputs\video_to_fbx\t1499_H4_ffffffffffff\final.fbx")
            try:
                FC.official_planted_ranges(OFFICIAL, joints_path=p)
                check("t999 npz + t1499 joints is refused", False, "no exception")
            except FC.TakeMismatchError as exc:
                check("t999 npz + t1499 joints is refused", True,
                      f"{pos_other.shape[0]} frames vs 995")
                print("      negative-case message:", str(exc))
            finally:
                os.remove(p)


def main() -> int:
    if not os.path.isfile(PROBE) or not os.path.isfile(OFFICIAL):
        return _skip(f"缺工件：probe={os.path.isfile(PROBE)} official={os.path.isfile(OFFICIAL)}")
    probe = np.load(PROBE, allow_pickle=True)
    part_axis(probe)
    part_guard(PROBE)
    print(f"==== {sum(checks)}/{len(checks)} PASS ====")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
