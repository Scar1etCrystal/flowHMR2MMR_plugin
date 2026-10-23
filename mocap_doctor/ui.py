import json

import bpy
from bpy.types import Panel, UIList

from . import annotation, project
from .presets import EXPECTED_FPS
from .workflow import STEP_INDEX, STEPS, step_at

# fix-row status → (icon, short label)。注意别用眼睛类图标——行首的静音
# 按钮就是眼睛，状态再画一只会被当成第二个按钮（用户实测踩过）。
_FIX_STATUS = {
    "preview": ("NONE", "预览"),
    "committed": ("CHECKMARK", "已提交"),
    "lost": ("ERROR", "丢失"),
    "unregistered": ("QUESTION", "未登记"),
}


class MD_UL_AgentFixes(UIList):
    """每条 agent 修复一行：静音眼睛 + 区间/工具 + 状态。"""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.scale_x = 1.0
        # 勾选框（普通复选框渲染；emboss=False 在未勾选时什么都不画）
        row.prop(item, "selected", text="")
        # 静音眼睛：单看这条修复贡献了什么
        mute = row.row(align=True)
        mute.emboss = "NONE"
        op = mute.operator(
            "mocap_doctor.agent_fix_mute",
            text="",
            icon="HIDE_ON" if item.muted else "HIDE_OFF",
        )
        op.op_id = item.op_id
        op.track = item.track
        op.strip = item.strip
        op.index = index

        status_icon, status_text = _FIX_STATUS.get(
            item.status, ("DOT", item.status or "?"))
        label = row.row(align=True)
        label.label(text=item.label or item.strip or "?")
        tail = label.row(align=True)
        tail.alignment = "RIGHT"
        tail.label(text=status_text, icon=status_icon)
        if abs(float(item.exponent) - 1.0) > 1e-6:
            tail.label(text=f"×{item.exponent:.2f}")

    def draw_filter(self, context, layout):
        pass

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        flt = [self.bitflag_filter_item] * len(items)
        return flt, []


def _range_count(scene, channels):
    return sum(len(annotation.get_channel_ranges(scene, channel)) for channel in channels)


def _draw_run(layout, step_id, text="运行预览", icon="PLAY"):
    operator = layout.operator("mocap_doctor.run_step", text=text, icon=icon)
    operator.step_id = step_id


def _draw_message(layout, message, icon="INFO", width=34):
    text = str(message or "")
    if not text:
        return
    for index in range(0, len(text), width):
        layout.label(text=text[index : index + width], icon=icon if index == 0 else "NONE")


class MD_PT_Main(Panel):
    bl_idname = "MD_PT_main"
    bl_label = "MoCap Doctor"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MoCap Doctor"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.mocap_doctor

        step = step_at(settings.current_step)
        # 常驻“指定区间”：测试时只跑这一段，不必等整段 3666 帧。
        range_row = layout.column(align=True)
        cells = range_row.row(align=True)
        cells.prop(settings, "mocap_frame_start", text="起")
        cells.prop(settings, "mocap_frame_end", text="止")
        range_row.operator("mocap_doctor.use_scene_range", text="用场景帧范围", icon="PREVIEW_RANGE")
        header = layout.column(align=True)
        header.label(text=f"{step.stage}  ·  {settings.current_step + 1}/{len(STEPS)}")
        header.label(text=step.label, icon="ANIM")
        record = project.find_step_record(settings, step.id, create=False)
        if (
            not settings.preview_step_id
            and record is not None
            and record.status in {"ACCEPTED", "STALE", "FAILED"}
        ):
            redo = header.operator(
                "mocap_doctor.restore_before_step",
                text=f"恢复到“{step.label}”执行前",
                icon="LOOP_BACK",
            )
            redo.step_id = step.id

        body = layout.column(align=True)
        body.enabled = not settings.busy and not settings.preview_step_id
        self.draw_step(context, body, step.id)

        if settings.preview_step_id:
            preview = layout.box()
            preview_record = project.find_step_record(
                settings, settings.preview_step_id, create=False
            )
            preview_failed = preview_record is not None and preview_record.status == "FAILED"
            if preview_failed:
                preview.alert = True
                preview.label(text="执行未完成", icon="ERROR")
                preview.operator(
                    "mocap_doctor.discard_preview",
                    text="丢弃失败结果并恢复",
                    icon="LOOP_BACK",
                )
            else:
                preview.label(text="等待人眼检查", icon="HIDE_OFF")
                row = preview.row(align=True)
                row.operator("mocap_doctor.accept_preview", icon="CHECKMARK")
                row.operator("mocap_doctor.discard_preview", icon="LOOP_BACK")

        if settings.status_message:
            feedback = layout.column(align=True)
            message = str(settings.status_message)
            if message.startswith('A ') and ' · B ' in message and ' · C ' in message:
                message = '修复已完成'
            icon = {"INFO": "CHECKMARK", "WARNING": "ERROR", "ERROR": "CANCEL"}.get(
                settings.status_level, "CHECKMARK"
            )
            _draw_message(feedback, message, icon=icon)

        self.draw_navigation(layout, settings)

    def draw_navigation(self, layout, settings):
        navigation = layout.row(align=True)
        navigation.enabled = not settings.preview_step_id and not settings.busy
        previous = navigation.operator(
            "mocap_doctor.navigate_step",
            text="上一步",
            icon="TRIA_LEFT",
        )
        previous.delta = -1
        navigation.label(text=f"浏览 {settings.current_step + 1}/{len(STEPS)}")
        following = navigation.operator(
            "mocap_doctor.navigate_step",
            text="下一步",
            icon="TRIA_RIGHT",
        )
        following.delta = 1

    def _draw_agent_body(self, layout, context, settings):
        """Agent 协作面板的内容：MD_PT_Agent 这个真子面板调用（默认折叠）。"""

        scene = context.scene
        agent = layout.column(align=True)
        try:
            from .core import agent_bridge
            st = agent_bridge.status()
            # draw 上下文禁止写 ID（改轨名/建 strip 会报 "Writing to ID
            # classes in this context is not allowed"）——重建与迁移全在
            # fixlist 定时器里做，这里只读。定时器万一没在跑（启动期注册
            # 丢失等）就从这里拉起：注册定时器不是 ID 写，draw 里允许。
            stale = (int(settings.agent_fixes_rev) != int(settings.agent_ops_rev)
                     or (len(settings.agent_fixes) == 0
                         and agent_bridge._rig_armature(settings, scene)
                         is not None))
            if stale:
                # 只保留自愈定时器：原来这里画一行“列表同步中…”，那只是实现细节，
                # 用户不想在面板上看见它。
                try:
                    agent_bridge.start_fixlist_timer()
                except Exception:
                    pass
            if st["running"]:
                agent.label(
                    text=f"服务运行中 127.0.0.1:{st['port']}  ·  v{st['version']}  ·  {st['clients']} 客户端",
                    icon="RADIOBUT_ON",
                )
                if st["last_tool"]:
                    agent.label(text=f"上次调用 {st['last_tool']}", icon="DOT")
                if st["last_error"]:
                    agent.label(text=f"错误：{st['last_error']}", icon="ERROR")
            else:
                agent.label(text="未运行", icon="RADIOBUT_OFF")
                # 错误行不只服务运行时才有意义——fixlist 定时器的异常也走这里
                if st["last_error"]:
                    agent.label(text=f"错误：{st['last_error']}", icon="ERROR")

            # ---- 修复列表 ----
            if settings.agent_fixes:
                agent.template_list(
                    "MD_UL_AgentFixes", "", settings, "agent_fixes",
                    settings, "agent_fix_index", rows=4,
                )
                # 选择工具行：全选 / 清空 / 选中同窗（一个发力窗的所有条目）
                srow = agent.row(align=True)
                sop = srow.operator("mocap_doctor.agent_fix_sel_all",
                                    text="全选", icon="CHECKBOX_HLT")
                sop.select = True
                sop = srow.operator("mocap_doctor.agent_fix_sel_all",
                                    text="清空", icon="CHECKBOX_DEHLT")
                sop.select = False
                srow.operator("mocap_doctor.agent_fix_sel_window",
                              text="选中同窗", icon="RESTRICT_SELECT_OFF")

                sel = [i for i in settings.agent_fixes if i.selected]
                index = min(settings.agent_fix_index,
                            len(settings.agent_fixes) - 1)
                item = settings.agent_fixes[index] if index >= 0 else None
                if sel:
                    # ---- 批量模式：滑块/眼睛/按钮作用于所有勾选行 ----
                    n = len(sel)
                    if any(not i.alive for i in sel):
                        agent.label(
                            text=f"选中里有 {sum(1 for i in sel if not i.alive)}"
                                 " 条场景数据已不在（跳过）", icon="ERROR")
                    row = agent.row(align=True)
                    row.label(text=f"力度({n}条)")
                    row.prop(settings, "agent_batch_exponent",
                             text="", slider=True)
                    row.prop(settings, "agent_batch_muted", text="",
                             icon="HIDE_ON" if settings.agent_batch_muted
                             else "HIDE_OFF")
                    buttons = agent.row(align=True)
                    buttons.operator("mocap_doctor.agent_fix_commit",
                                     text=f"提交({n})", icon="CHECKMARK")
                    buttons.operator("mocap_doctor.agent_fix_revert",
                                     text=f"撤销({n})", icon="LOOP_BACK")
                    buttons.operator("mocap_doctor.agent_fix_forget",
                                     text=f"清除({n})", icon="TRASH")
                elif item is not None:
                    if item.status in ("lost", "unregistered"):
                        agent.label(
                            text="这条修复的场景数据不在了（记录保留）",
                            icon="ERROR")
                    row = agent.row(align=True)
                    row.label(text="力度")
                    row.prop(item, "exponent", text="", slider=True)
                    row.prop(item, "muted", text="",
                             icon="HIDE_ON" if item.muted else "HIDE_OFF")
                    buttons = agent.row(align=True)
                    op = buttons.operator("mocap_doctor.agent_fix_commit",
                                          text="提交", icon="CHECKMARK")
                    op.op_id = item.op_id
                    op = buttons.operator("mocap_doctor.agent_fix_revert",
                                          text="撤销", icon="LOOP_BACK")
                    op.op_id = item.op_id
                    op = buttons.operator("mocap_doctor.agent_fix_forget",
                                          text="清除记录", icon="TRASH")
                    op.op_id = item.op_id
                    # ---- 参数控件：AI 把可调的暴露出来，人拖滑块就改 ----
                    # 连续值防抖 250ms 同轨重写；离散选项/方向物体点完即写。
                    params = [p for p in settings.agent_params
                              if p.op_id == item.op_id]
                    if params:
                        pbox = agent.box()
                        pbox.label(text="参数（改完自动重写，不用重聊）",
                                   icon="PREFERENCES")
                        for p in params:
                            prow = pbox.row(align=True)
                            if p.kind == "float":
                                prow.prop(p, "fval", text=p.label,
                                          slider=True)
                            elif p.kind == "int":
                                prow.prop(p, "ival", text=p.label)
                            elif p.kind == "range":
                                prow.label(text="帧范围")
                                prow.prop(p, "ival", text="起")
                                prow.prop(p, "ival2", text="止")
                            elif p.kind == "object":
                                prow.prop(p, "obj", text=p.label)
                                nb = prow.operator(
                                    "mocap_doctor.agent_dir_create",
                                    text="", icon="ADD")
                                nb.kind = "arrow"
                                nb.assign = p.pkey
                                nb = prow.operator(
                                    "mocap_doctor.agent_dir_create",
                                    text="", icon="TRACKER")
                                nb.kind = "aim"
                                nb.assign = p.pkey
                            elif p.kind == "choice":
                                prow.label(text=p.label)
                                try:
                                    opts = json.loads(p.options or "[]")
                                except Exception:
                                    opts = []
                                for opt in opts:
                                    chosen = (p.sval == str(opt))
                                    b = prow.operator(
                                        "mocap_doctor.agent_param_choice",
                                        text=str(opt),
                                        icon=("RADIOBUT_ON" if chosen
                                              else "RADIOBUT_OFF"))
                                    b.pkey = p.pkey
                                    b.value = str(opt)
                            else:
                                prow.label(text=f"{p.label}: {p.sval}")
            else:
                agent.label(text="暂无修复", icon="DOT")

            row = agent.row(align=True)
            row.operator("mocap_doctor.agent_server_toggle",
                         text="停止服务" if st["running"] else "启动服务",
                         icon="CANCEL" if st["running"] else "PLAY")
            row.operator("mocap_doctor.agent_ab_toggle",
                         text="A/B 全部", icon="HIDE_OFF")
            # 刷新常驻（列表为空时也要能手动触发迁移+重建——定时器失灵的兜底）
            row.operator("mocap_doctor.agent_fix_refresh",
                         text="", icon="FILE_REFRESH")
        except Exception as exc:
            agent.label(text=f"Agent 模块加载失败：{exc}", icon="ERROR")

    def draw_step(self, context, layout, step_id):
        scene = context.scene
        settings = scene.mocap_doctor

        if step_id == "import_official":
            # 第 0 步：一句提示 + 一个参数 + 一个按钮，别的一律没有。
            layout.label(text="请使用空文件导入动作")
            layout.prop(settings, "official_take_dir", text="动作数据目录")
            _draw_run(layout, step_id, "导入", "IMPORT")

        elif step_id == "foot_spacing":
            # 脚间距补偿：排在重定向之后；参数就画在这一步的参数区里（不再另起面板）。
            layout.prop(settings, "foot_spacing_clearance_ratio", text="贴紧间距比")
            layout.prop(settings, "foot_spacing_close_fade_ratio", text="淡出间距比")
            layout.prop(settings, "foot_spacing_crossing_fade", text="交叉淡出（帧）")
            layout.prop(settings, "foot_spacing_crossing_detect", text="交叉判定窗（帧）")
            layout.prop(settings, "foot_spacing_crossing_min_gap", text="交叉最小间距比")
            _draw_run(layout, step_id, "运行补偿", "PLAY")

        elif step_id == "retarget":
            layout.prop(settings, "retarget_mode", text="模式")
            if settings.retarget_mode == "AUTO":
                layout.prop(settings, "retarget_route", text="路线")
                if getattr(settings, "retarget_route", "MMR") == "MMD":
                    layout.prop(settings, "source_armature", text="源骨架")
                    layout.prop(settings, "mmd_armature", text="MMD 骨架")
                else:
                    layout.prop(settings, "source_armature", text="源骨架")
                    layout.prop(settings, "mmr_rig", text="目标骨架")
                    layout.prop(settings, "retarget_bmap_choice", text="映射")
                    if settings.retarget_bmap_choice == "CUSTOM":
                        layout.prop(settings, "retarget_bmap_custom", text="映射文件")
                    layout.prop(settings, "retarget_preset", text="rest 预设")
                _draw_run(layout, step_id, "重定向", "MOD_ARMATURE")
            else:
                # 手动模式：插件不介入，只等你在外面做完。
                layout.prop(settings, "mmr_rig", text="目标骨架")
                layout.operator("mocap_doctor.mark_retarget_done",
                                text="重定向已完成", icon="CHECKMARK")

        elif step_id == 'hand_spacing':
            layout.prop(settings, 'hand_spacing_strength', text='补偿强度')
            _draw_run(layout, step_id, '手间距补偿', 'MOD_ARMATURE')
            layout.operator('mocap_doctor.finger_align', text='手指校正', icon='BONE_DATA')

        elif step_id == 'contacts':
            # 贴地区间：数据由导入步骤接好，内部自动决定来源；
            # 这一页只留 检测 / 编辑 / 计数 这些功能。
            layout.label(text="贴地区间", icon="SORTTIME")
            planted_row = layout.row(align=True)
            _draw_run(planted_row, "contacts", "检测", "VIEWZOOM")
            editor = planted_row.operator(
                "mocap_doctor.enter_annotation_mode", text="编辑", icon="NLA"
            )
            editor.channel_group = "FOOT"
            planted_row.operator(
                "mocap_doctor.reset_effective_contacts", text="", icon="FILE_REFRESH"
            )
            count = _range_count(
                scene,
                (annotation.CHANNEL_FOOT_L_EFFECTIVE, annotation.CHANNEL_FOOT_R_EFFECTIVE),
            )
            layout.label(text=f"当前贴地区间 {count} 个")

        elif step_id == "target_floor":
            layout.prop(settings, "target_mesh", text="模型网格")
            layout.prop(settings, "target_floor_z", text="地面 Z")
            layout.prop(settings, "target_floor_strength", text="强度")
            layout.prop(settings, "target_floor_tolerance", text="容差")
            layout.prop(settings, "target_clearance", text="离地间隙")
            layout.prop(settings, "target_floor_max_lift", text="单帧最大抬升")
            layout.prop(settings, "target_floor_smooth_radius", text="平滑半径")
            layout.prop(settings, "target_floor_max_delta", text="帧间最大变化（m/帧）")
            layout.prop(settings, "target_vertex_sample_step", text="顶点采样步长")
            layout.prop(settings, "target_visible_only", text="只扫描可见 Mesh")
            layout.prop(settings, "target_reset_existing_z", text="重建校正 Z 曲线")
            _draw_run(layout, step_id)

        elif step_id == "foot_lock":
            layout.prop(settings, "lock_min_xy_range", text="最小修复漂移")
            layout.prop(settings, "lock_anchor_mode", text="锚点")
            layout.prop(settings, "lock_blend_frames", text="缓入缓出")
            layout.prop(settings, "lock_trim", text="裁掉段首尾")
            layout.prop(settings, "drift_min_segment_len", text="漂移分析最短区间")
            layout.prop(settings, "lock_min_segment_len", text="锁定最短区间")
            layout.prop(settings, "pelvis_correction_max", text="骨盆修约上限")
            row = layout.row(align=True)
            row.label(text="锁位置")
            row.prop(settings, "lock_x", text="X", toggle=True)
            row.prop(settings, "lock_y", text="Y", toggle=True)
            row.prop(settings, "lock_z", text="Z", toggle=True)
            _draw_run(layout, step_id)

        elif step_id == "ground_feet":
            # 整个"贴地锁定"步骤原样搬过来（含腾空标注编辑）；长说明改成按钮的
            # 悬停提示（mocap_doctor.enter_annotation_air 的 bl_description）。
            layout.prop(settings, "target_floor_z", text="地面 Z")
            layout.prop(settings, "target_clearance", text="离地间隙")
            layout.prop(settings, "ground_smooth_radius", text="平滑半径")
            layout.prop(settings, "ground_max_delta", text="帧间最大变化（m/帧）")
            operator = layout.operator(
                "mocap_doctor.enter_annotation_air",
                text="编辑腾空区间",
                icon="NLA",
            )
            layout.operator(
                "mocap_doctor.reload_air_hints",
                text="载入自动腾空提示",
                icon="FILE_REFRESH",
            )
            _draw_run(layout, step_id, "贴地锁定", "SNAP_ON")

        elif step_id == "mmd_bake":
            layout.prop(settings, "mmd_armature", text="原生 MMD 骨架")
            layout.prop(settings, "mmr_rig", text="MMR Rig")
            layout.prop(settings, "mmd_bake_mode", text="Bake 策略")
            layout.operator("mocap_doctor.mmd_bake", text="自动烘焙", icon="ACTION")
            layout.operator("mocap_doctor.validate_manual_mmd_bake",
                            text="检测并清理手动烘焙结果", icon="CHECKMARK")


        elif step_id == "export_prep":
            layout.prop(settings, "vmd_floor_offset", text="VMD 地面 Z 偏移")
            reminder = layout.box()
            reminder.label(text="MMR 手部导出清理", icon="ARMATURE_DATA")
            reminder.label(text="确认上肢已烘到“腕 / 肘 / 手首”后再清理。")
            reminder.label(text="再恢复肘部层级，删除 Blender 编辑用手 IK。")
            reminder.label(text="只修改这份工作文件；确认后会创建恢复检查点。")
            layout.operator("mocap_doctor.prepare_vmd_export", text="确认并生成 VMD 导出预览", icon="MODIFIER")

        elif step_id == "export":
            layout.label(text="确认 MMR 约束已禁用，隐藏 RIG 后动作仍正常")
            layout.prop(settings, "vmd_export_path", text="VMD 文件")
            layout.operator("mocap_doctor.export_vmd", text="导出 VMD", icon="EXPORT")

        else:
            self.draw_step_deprecated(context, layout, step_id)

    def draw_step_deprecated(self, context, layout, step_id):
        """**DEPRECATED**：已经从向导移除的步骤（Q6：只从 UI 移除，代码保留）。

        这些分支现在够不到（STEPS 里没有这些 id）；留着是因为它们背后的运行器
        （RUN_STEP_HANDLERS）和参数都还在，将来要恢复某一步时不必重写。
        其中 contacts 的"检测 planted / 编辑 planted"两个按钮搬到了 retarget 页。
        """

        scene = context.scene
        settings = scene.mocap_doctor

        if step_id == "source_check":
            layout.prop(settings, "source_armature", text="源骨架")
            layout.prop(settings, "source_profile", text="源数据来源")
            layout.label(text="仅处理上方有效动捕范围")
            _draw_run(layout, step_id, "校验源数据", "CHECKMARK")

        elif step_id == "source_floor":
            layout.prop(settings, "source_floor_z", text="地面 Z")
            layout.prop(settings, "source_floor_strength", text="强度")
            layout.prop(settings, "source_floor_window", text="地面估计窗口（帧）")
            layout.prop(settings, "source_floor_tolerance", text="容差")
            layout.prop(settings, "source_floor_clearance", text="离地间隙")
            layout.prop(settings, "source_floor_max_lift", text="单帧最大抬升")
            layout.prop(settings, "source_floor_smooth_radius", text="平滑半径")
            layout.prop(settings, "source_floor_max_delta", text="帧间最大变化（m/帧）")
            _draw_run(layout, step_id)

        elif step_id == "contacts":
            layout.prop(settings, "contact_height", text="接触高度")
            layout.prop(settings, "contact_xy_speed", text="XY 速度阈值（m/帧）")
            layout.prop(settings, "contact_moving_xy_speed", text="移动判定速度（m/帧）")
            layout.prop(settings, "contact_vertical_speed", text="垂直速度阈值（m/帧）")
            layout.prop(settings, "contact_anchor_drift", text="最大锚点漂移")
            layout.prop(settings, "contact_min_segment_len", text="最短区间")
            layout.prop(settings, "contact_merge_gap", text="合并间隔")
            layout.prop(settings, "contact_penetration_tolerance", text="穿地诊断容差")
            _draw_run(layout, step_id, "重新检测 Planted", "VIEWZOOM")
            count = _range_count(
                scene,
                (annotation.CHANNEL_FOOT_L_EFFECTIVE, annotation.CHANNEL_FOOT_R_EFFECTIVE),
            )
            layout.label(text=f"当前有效区间 {count} 个")
            operator = layout.operator(
                "mocap_doctor.enter_annotation_mode",
                text="编辑 planted 区间",
                icon="NLA",
            )
            operator.channel_group = "FOOT"
            layout.operator("mocap_doctor.reset_effective_contacts", icon="FILE_REFRESH")

        elif step_id == "global_correction":
            layout.prop(settings, "model_root", text="模型根对象")
            layout.prop(settings, "mmr_rig", text="MMR Rig")
            layout.prop(settings, "global_rot_x", text="旋转 X")
            layout.prop(settings, "global_rot_y", text="旋转 Y")
            layout.prop(settings, "global_rot_z", text="旋转 Z")
            _draw_run(layout, step_id)

        elif step_id == "tilt":
            layout.prop(settings, "tilt_reference_frame", text="参考帧")
            layout.prop(settings, "tilt_strength", text="强度")
            row = layout.row(align=True)
            row.prop(settings, "tilt_damp_x", text="X", toggle=True)
            row.prop(settings, "tilt_damp_y", text="Y", toggle=True)
            row.prop(settings, "tilt_damp_z", text="Z", toggle=True)
            _draw_run(layout, step_id)


class MD_PT_Agent(Panel):
    """Agent 协作：真子面板（原来是主面板里的一个 box），默认折叠。

    Q9：默认折叠 + 记住展开状态。折叠状态存在 scene 属性 agent_panel_open 上，
    随 .blend 保存，所以下次打开同一份工程还是上次的样子。
    """

    bl_idname = "MD_PT_agent"
    bl_label = "Agent 协作"
    bl_parent_id = "MD_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MoCap Doctor"

    @classmethod
    def poll(cls, context):
        return context.scene.mocap_doctor.initialized

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mocap_doctor
        header = layout.row(align=True)
        header.prop(
            settings,
            "agent_panel_open",
            text="",
            icon="TRIA_DOWN" if settings.agent_panel_open else "TRIA_RIGHT",
            emboss=False,
        )
        header.label(text="Agent 协作", icon="CONSOLE")
        if not settings.agent_panel_open:
            return
        MD_PT_Main._draw_agent_body(self, layout, context, settings)


class MD_PT_Project(Panel):
    bl_idname = "MD_PT_project"
    bl_label = "项目与对象"
    bl_parent_id = "MD_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MoCap Doctor"

    @classmethod
    def poll(cls, context):
        return context.scene.mocap_doctor.initialized

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mocap_doctor
        layout.prop(settings, "source_armature", text="源骨架")
        layout.prop(settings, "model_root", text="模型根对象")
        layout.prop(settings, "target_mesh", text="模型网格")
        layout.prop(settings, "mmr_rig", text="MMR Rig")
        layout.prop(settings, "mmd_armature", text="MMD Armature")
        layout.prop(settings, "correction_empty", text="校正 Empty")
        layout.operator("mocap_doctor.discover_objects", icon="VIEWZOOM")
        layout.operator("mocap_doctor.create_checkpoint", icon="BOOKMARKS")
        restore = layout.row()
        restore.enabled = bool(settings.last_checkpoint)
        restore.operator("mocap_doctor.restore_last_checkpoint", icon="LOOP_BACK")
        layout.separator()
        layout.operator("mocap_doctor.delete_checkpoints",
                        text="删除所有检查点", icon="TRASH")


class MD_PT_NLAAnnotation(Panel):
    bl_idname = "MD_PT_nla_annotation"
    bl_label = "MoCap Doctor 标注"
    bl_space_type = "NLA_EDITOR"
    bl_region_type = "UI"
    bl_category = "MoCap Doctor"

    @classmethod
    def poll(cls, context):
        return bool(getattr(context.scene, "mcd_annotation_mode", False))

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        active_channel = annotation._context_track_channel(context) or scene.mcd_annotation_active_channel
        channel_row = layout.row(align=True)
        for definition in annotation.visible_track_definitions(scene):
            operator = channel_row.operator(
                "mcd.annotation_set_channel",
                text=definition["label"],
                depress=active_channel == definition["channel"],
            )
            operator.channel = definition["channel"]
        layout.label(text=f"当前帧 {scene.frame_current}")
        if scene.mcd_annotation_has_pending_in:
            pending = layout.box()
            label = annotation.TRACK_BY_CHANNEL[scene.mcd_annotation_pending_channel]["label"]
            pending.label(
                text=f"入点 {scene.mcd_annotation_pending_in} · {label}",
                icon="MARKER_HLT",
            )
            pending.operator("mcd.annotation_cancel_in", icon="X")
        row = layout.row(align=True)
        row.operator("mcd.annotation_mark_in", text="标记 / 覆盖入点", icon="PREV_KEYFRAME")
        row.operator("mcd.annotation_mark_out", text="标记 / 覆盖出点", icon="NEXT_KEYFRAME")
        layout.operator("mcd.annotation_split_current", text="在当前帧拆分", icon="MOD_EDGESPLIT")
        row = layout.row(align=True)
        row.operator("mcd.annotation_delete", text="删除所选区间", icon="TRASH")
        row.operator("mcd.annotation_locate", text="跳到所选区间", icon="VIEWZOOM")
        layout.operator("mcd.annotation_loop_selected", text="循环检查所选区间", icon="PLAY")
        layout.operator("mocap_doctor.exit_annotation_mode", icon="TIME")


CLASSES = (MD_PT_Main, MD_PT_Agent, MD_PT_Project,
           MD_PT_NLAAnnotation, MD_UL_AgentFixes)


def register_ui():
    registered = []
    try:
        for cls in CLASSES:
            bpy.utils.register_class(cls)
            registered.append(cls)
    except Exception:
        for cls in reversed(registered):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
        raise


def unregister_ui():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
