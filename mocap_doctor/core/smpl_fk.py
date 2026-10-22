"""SMPL forward kinematics in numpy (absolute joint positions).

Written for `F:\\flowhmr\\neutral\\model.npz` (full 52-joint SMPL: shapedirs 16
betas, posedirs (6890,3,459), J_regressor (52,6890), kintree_table (2,52)).
Pose blend shapes are deliberately not used: they only move the MESH; the joints
come from the rigid accumulation of the kinematic tree (the official code uses
``_forward_smpl_joints``, `postprocess_utils.py:67`, for exactly this).

Absolute joints = A_j @ J_j + trans, with A_j = A_parent @ T_j.
"""

from __future__ import annotations

from typing import Any

_MODEL_CACHE: dict[str, Any] = {}
FPS = 30.0


def load_model(model_path: str) -> dict[str, Any]:
    import numpy as np

    key = str(model_path)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with np.load(key, allow_pickle=True) as data:
        model = {
            "v_template": np.asarray(data["v_template"], dtype=np.float64),
            "shapedirs": np.asarray(data["shapedirs"], dtype=np.float64),
            "J_regressor": np.asarray(data["J_regressor"], dtype=np.float64),
            "kintree_table": np.asarray(data["kintree_table"]),
        }
    parents = [int(p) for p in model["kintree_table"][0]]
    parents[0] = -1
    model["parents"] = parents
    _MODEL_CACHE[key] = model
    return model


def _rodrigues(rotvec: Any) -> Any:
    import numpy as np

    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-12, 1.0, theta)
    axis = rotvec / safe
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    K = np.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1
    ).reshape(rotvec.shape[:-1] + (3, 3))
    ident = np.eye(3)
    t = theta[..., None]
    return ident + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def forward_joints(
    model_path: str, poses: Any, betas: Any, trans: Any
) -> Any:
    """(T,52,3) axis-angle poses + betas + trans -> (T,52,3) absolute joints."""

    import numpy as np

    model = load_model(model_path)
    pose = np.asarray(poses, dtype=np.float64)
    if pose.ndim == 2:  # single frame
        pose = pose[None]
    beta = np.asarray(betas, dtype=np.float64).reshape(-1)[: model["shapedirs"].shape[2]]
    tr = np.asarray(trans, dtype=np.float64).reshape(-1, 3)
    T = pose.shape[0]
    if tr.shape[0] == 1 and T > 1:
        tr = np.repeat(tr, T, axis=0)

    rotation = _rodrigues(pose)                       # (T,J,3,3)
    v_shaped = model["v_template"] + model["shapedirs"] @ beta
    joints_rest = model["J_regressor"] @ v_shaped     # (J,3)
    J = rotation.shape[1]

    local = np.zeros((T, J, 4, 4))
    local[:, :, :3, :3] = rotation
    local[:, :, :3, 3] = joints_rest[None, :, :]
    local[:, :, 3, 3] = 1.0

    out = np.zeros_like(local)
    out[:, 0] = local[:, 0]
    order = list(range(J))
    for j in order[1:]:
        parent = model["parents"][j]
        if parent < 0:
            out[:, j] = local[:, j]
        else:
            out[:, j] = out[:, parent] @ local[:, j]
    return out[:, :, :3, 3] + tr[:, None, :]


def forward_joints_from_file(model_path: str, npz_path: str) -> Any:
    """Convenience: read poses/betas/trans from an npz and FK them."""

    import numpy as np

    with np.load(str(npz_path), allow_pickle=True) as data:
        poses = np.asarray(data["poses"], dtype=np.float64)
        betas = np.asarray(data["betas"], dtype=np.float64)
        if "trans" in data:
            trans = np.asarray(data["trans"], dtype=np.float64)
        else:
            trans = np.asarray(data["smooth_root_pos"], dtype=np.float64)
    return forward_joints(model_path, poses, betas, trans)


if __name__ == "__main__":  # self-test against the official take output
    import numpy as np

    MODEL = r"F:\flowhmr\neutral\model.npz"
    REF = r"F:\flowhmr_official_out\t999_OFFICIAL_post.npz"
    joints = forward_joints_from_file(MODEL, REF)
    ref = np.load(REF, allow_pickle=True)
    local_ref = np.asarray(ref["local_joints_positions"], dtype=np.float64)
    rel = joints - joints[:, :1, :]
    ids = {"L_Ankle": 7, "L_Foot": 10, "R_Ankle": 8, "R_Foot": 11}
    print("FK joints", joints.shape, "vs file local_joints", local_ref.shape)
    for name, idx in ids.items():
        err = np.abs(rel[:, idx, :] - (local_ref[:, idx, :] - local_ref[:, 0:1, :])).max()
        print(f"  {name:8} rel compare max err {err * 100:.2f} cm | abs Y min "
              f"{joints[:, idx, 1].min():.4f}")
    print("  abs Y min over all 52 joints:", round(float(joints[..., 1].min()), 4))
