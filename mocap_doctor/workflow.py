from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    label: str
    stage: str
    mutates_scene: bool
    description: str


# 向导只显示用户保留的步骤（2026-10 大改；2026-10-16 结构定稿）。
# 离开 UI 的步骤代码还在（RUN_STEP_HANDLERS + 旧页面），只是导航里到不了，
# 见 RETIRED_STEPS。
#
# 顺序（用户 2026-10-16 逐条指定；晚间删除“创建工作副本”页）：
#   index 0 = 导入（工作副本在第一次导入时自动建立，不再是单独一页）
#   index 1 = 重定向（结束时自动修复骨架）；index 2 = 贴地区间
#   手、脚间距补偿均排在重定向之后，各自保留可比较的预览
STEPS = (
    WorkflowStep(
        "import_official",
        "导入",
        "导入",
        True,
        "一键导入动作骨架与参考视频（自动找数据、找视频，找不到时给一句提示）",
    ),
    WorkflowStep(
        "retarget",
        "重定向",
        "重定向",
        True,
        "自动重定向并修复目标骨架，完成后统一检查预览",
    ),
    WorkflowStep("contacts", "贴地区间", "重定向", False, "检测、编辑源动作的贴地区间"),
    WorkflowStep("hand_spacing", "手间距补偿", "目标模型", True, "保持手腕方向，修复近手动作的FK间距"),
    WorkflowStep(
        "foot_spacing",
        "脚间距补偿",
        "目标模型",
        True,
        "两脚靠得很近时把目标脚间距拉向 rest 间距（写 IK）；排在重定向之后",
    ),
    WorkflowStep("ground_feet", "贴地锁定", "目标模型", True, "只在贴地区间把最低脚钉到地面，腾空段弹道重建"),
    WorkflowStep("foot_lock", "脚滑修复", "目标模型", True, "贴地区间内的脚位漂移分析与 XY 锁定"),
    WorkflowStep("target_floor", "模型穿地修复", "目标模型", True, "把网格最低点放回地面"),
    WorkflowStep("mmd_bake", "MMD Visual Bake", "导出", True, "安全检测后自动 Bake"),
    WorkflowStep("export_prep", "VMD 导出准备", "导出", True, "烘根补偿、删腿 FK、加地面偏移"),
    WorkflowStep("export", "导出 VMD", "导出", False, "调用 mmd_tools 导出"),
)

STEP_INDEX = {step.id: index for index, step in enumerate(STEPS)}

# 不在向导里、但代码仍保留执行器/页面的步骤（2026-10 精简 + 2026-10-16 收编预留槽）。
RETIRED_STEPS = (
    "project",
    "source_check",
    "source_floor",
    "target_skeleton_fix",
    "global_correction",
    "tilt",
)

# 不需要先建项目就能跑的步骤（导入本身不依赖项目状态）。
PROJECT_FREE_STEPS = ("import_official",)


def is_project_free_step(step_id):
    return str(step_id or "") in PROJECT_FREE_STEPS


def is_retired_step(step_id):
    return str(step_id or "") in RETIRED_STEPS


def clamp_step(index):
    return max(0, min(int(index), len(STEPS) - 1))


def step_at(index):
    return STEPS[clamp_step(index)]
