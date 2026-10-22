"""力度旋钮对 Euler 通道必须按【旋转】缩放，不是逐分量线性缩放。

2026-10-06 事故：一条 motion_copy 的 forearm_fk.L delta 达 179.41°，力度拖到 0.137
后逐帧 ±20° 乱摆、手腕冒出 57/55 mm 位置尖峰。

失效机制是 **Euler 环绕**：`+179.9°` 和 `−179.9°` 作为【旋转】只差 0.2°，
逐分量线性乘 0.1374 会把它们拉成相差 **49°** 的两个值 —— 把 0.2° 的抖动放大 246 倍。
旋转空间缩放（轴角 × exponent）不会。

单轴 delta 下两种做法**完全等价**（(179°,0,0)×0.5 本来就是半转），所以判别用例必须
要么是多轴、要么是环绕这一对。
"""
import math
import sys

import addon_utils
import bpy

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_ops  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


def q_of(euler_rad, order="XYZ"):
    from mathutils import Euler
    return Euler(list(euler_rad), order).to_quaternion()


def q_gap(qa, qb):
    """两个旋转之间的夹角（度）——这才是"看起来差多少"。"""
    return math.degrees(2.0 * math.acos(min(1.0, abs(qa.dot(qb)))))


def ang_of(euler_rad, order="XYZ"):
    return math.degrees(q_of(euler_rad, order).angle)


R = math.radians

# ── 多轴 delta：轴不变、角度成比例 ─────────────────────────────────────
DELTA = (120.0, 60.0, -30.0)
d_r = [R(v) for v in DELTA]
a0 = ang_of(d_r)
new = agent_ops._scale_euler_delta(d_r, 0.5, "XYZ")
q0, qn = q_of(d_r), q_of(new)
axis_gap = q_gap(q0, qn)
check("多轴 delta ×0.5 → 转角减半",
      abs(ang_of(new) - a0 / 2.0) < 0.5,
      f"原 {a0:.2f}° → {ang_of(new):.2f}°（期望 {a0/2:.2f}°）")

# ── 真正的失效机制：Euler 环绕 ────────────────────────────────────────
E = 0.13740456
POS, NEG = (179.9, 0.0, 0.0), (-179.9, 0.0, 0.0)
pos_r, neg_r = [R(v) for v in POS], [R(v) for v in NEG]
gap_before = q_gap(q_of(pos_r), q_of(neg_r))
gap_new = q_gap(q_of(agent_ops._scale_euler_delta(pos_r, E, "XYZ")),
                q_of(agent_ops._scale_euler_delta(neg_r, E, "XYZ")))
gap_old = q_gap(q_of([v * E for v in pos_r]), q_of([v * E for v in neg_r]))
check("环绕：两帧本来只差 0.2°（前提成立）",
      0.1 < gap_before < 0.5, f"原始旋转夹角 {gap_before:.3f}°")
check("环绕【已知极限】：两种实现都救不了，所以代码必须报警而不是静默缩放",
      gap_old > 10.0 and gap_new > 10.0,
      f"新 {gap_new:.2f}° / 旧 {gap_old:.2f}°（均为 {gap_before:.3f}° 的 "
      f"{gap_new/max(gap_before,1e-9):.0f}×）——R^t 在 π 处不连续，换表示也无效；"
      f"正确做法是写 delta 时不翻面")
check("护栏常量存在（>170° 的 Euler delta 触发警告）",
      hasattr(agent_ops, "_EXP_NEAR_PI") and abs(agent_ops._EXP_NEAR_PI - 2.967) < 0.01,
      f"_EXP_NEAR_PI = {getattr(agent_ops, '_EXP_NEAR_PI', None)}")

# ── exponent = 1 恒等（mathutils 是 float32，容差 1e-6；真实路径里 exponent==1 已短路）──
one = agent_ops._scale_euler_delta(d_r, 1.0, "XYZ")
check("exponent=1 → 恒等（float32 容差内）",
      all(abs(one[i] - d_r[i]) < 1e-6 for i in range(3)),
      f"最大偏差 {max(abs(one[i]-d_r[i]) for i in range(3)):.2e} rad")

# ── exponent = 0 零旋转 ───────────────────────────────────────────────
check("exponent=0 → 零旋转（静音）",
      ang_of(agent_ops._scale_euler_delta(d_r, 0.0, "XYZ")) < 1e-4, "")

# ── 事故量级 ──────────────────────────────────────────────────────────
BIG = [R(179.41), 0.0, 0.0]
nb = agent_ops._scale_euler_delta(BIG, E, "XYZ")
check("事故量级 179.41°×0.1374 → 24.66°",
      abs(ang_of(nb) - 179.41 * E) < 0.05, f"→ {ang_of(nb):.2f}°")

# ── 六种 order 自洽 ───────────────────────────────────────────────────
ok_orders = {}
for order in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
    a_o = ang_of(d_r, order)
    v = agent_ops._scale_euler_delta(d_r, 0.37, order)
    ok_orders[order] = abs(ang_of(v, order) - a_o * 0.37) < 0.5
check("六种 Euler order 都自洽", all(ok_orders.values()), str(ok_orders))

bad = [n for n, ok in RESULTS if not ok]
print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} PASS ====")
for n in bad:
    print("FAILED:", n)
if bad:
    sys.exit(1)
