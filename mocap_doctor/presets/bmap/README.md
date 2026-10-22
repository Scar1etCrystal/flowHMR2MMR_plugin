# ARP 重定向映射（.bmap）

给 PoseCapture_Pack 的「自定义映射」用的骨骼映射文件。格式是 Auto-Rig Pro 的
bmap：每条记录 5 行（目标骨%参数 / 源骨 / set_as_root / ik / ik_pole），ARP 用
`readlines()` 读取，行数必须是 5 的整数倍。

| 文件 | 用途 |
|---|---|
| `gvhmr_mmd_fix_A_foot_ik.bmap` | **默认**。GVHMR(SMPL 源) → Teto 的 MMR 控制骨架。脚部映射与已实测的 FreeMoCap→MMR 映射同构：`foot_ik.L ← f_avg_L_Ankle`（`ik=True`，pole=`thigh_ik_target.L`） |
| `gvhmr_mmd_fix_B_drop_foot_fk.bmap` | 回退。只删掉两条错误的脚部行，保留 FK 腿（脚踝不能独立活动） |
| `freemocap_mmd_reference.bmap` | 存档参考：早期实测可用的 FreeMoCap→MMR 映射，A 版的脚部写法来自它 |

## 为什么需要修正版

PoseCapture 内置的 MMD 预设（`_BMAP_MMD_FEMALE`）在 arue 式 Teto + 该 MMR 版本上：

- `foot_ik.L/R`（真正的 IK 控制骨）**完全没有被映射**；
- 唯一带 `ik=True` + pole 的行挂在了 `foot_spin_ik.L/R`（IK 链里的旋转手柄）上；
- 脚踝走 `foot_fk.L ← f_avg_L_Ankle` 的 FK 纯旋转行。

后果（实测，世界空间脚踝俯仰角，2290 帧舞蹈片段）：脚部姿态与源数据平均相差
**+67.96°（左）/ +66.56°（右）**，表现为"脚尖固定翘起、脚掌心朝前"。

修正版把脚踝改由 `foot_ik` 驱动（IK 解算），并要求把腿的 IK/FK 滑条设为全 IK。

**这一条对模型绑定敏感**：换任何新的 MMD 模型 / MMR 版本，脚部映射都要重新验证
（用世界空间的脚踝俯仰角与源数据对比，别看局部四元数——跨骨架不可比）。

## 用法（PoseCapture 面板）

1. 目标骨架 = Teto 的 MMR 控制骨架（`RIG-..._arm`）
2. 控制器类型 = **自定义映射**，映射文件 = 本目录的 `gvhmr_mmd_fix_A_foot_ik.bmap`
3. 点一次「预处理控制器」（设置骨层与 IK/FK 滑条），**然后**把
   `thigh_parent.L/R` 的 `IK_FK` 从 0.95 改成 **0**（1=FK，0=IK）
   ——顺序不能反，预处理会把值写回 0.95
4. 「重定向到 ARP」

注意：控制器类型选「自定义映射」时，PoseCapture 的「预处理控制器」按钮会隐藏
（需先切到 MMD 类型跑一次），其「修复脚底滑动」也会走通用骨骼名回退分支而报错
——脚滑另走我们的 foot_lock 或临时切回 MMD 类型。
