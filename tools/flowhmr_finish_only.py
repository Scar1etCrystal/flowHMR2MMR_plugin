"""仅收尾：隐藏 FlowHMR 源骨架/参考网格并报告目标动作。
不运行 ARP、不重新校准 rest、不重复脚距补偿、不保存文件。
在目前已经跑完重定向的 Blender 场景中运行即可。
"""
import bpy

SOURCE_ARM = 'Reference'
TARGET_ARM = 'RIG-arue式重音テトver 2.01_arm'


def hide_source_safely(src):
    vl = bpy.context.view_layer
    for ob in [src] + [c for c in src.children if c.type == 'MESH']:
        if ob.name in vl.objects:
            ob.hide_set(True, view_layer=vl)
            viewport = '已隐藏'
        else:
            # 此对象本来就不在本层，无需也不能调用 hide_set。
            viewport = '不在当前 View Layer，跳过 hide_set（本层不可见）'
        ob.hide_render = True
        print(f'[收尾] {ob.name}: {viewport}；hide_render=True')


def main():
    src = bpy.data.objects.get(SOURCE_ARM)
    tgt = bpy.data.objects.get(TARGET_ARM)
    if src is None or tgt is None or src.type != 'ARMATURE' or tgt.type != 'ARMATURE':
        raise RuntimeError('源或目标骨架不存在，请检查脚本开头的名称')
    ad = tgt.animation_data
    act = ad.action if ad else None
    if act is None:
        raise RuntimeError('目标 RIG 没有活动 Action；不把收尾当作重定向成功，请检查 NLA/日志')
    print(f'[目标] {tgt.name} / Action={act.name} / 帧范围={tuple(act.frame_range)}')
    try:
        fcs = act.fcurves
        print(f'[目标] FCurve={len(fcs)}，关键帧={sum(len(fc.keyframe_points) for fc in fcs)}')
    except (AttributeError, RuntimeError):
        print('[目标] 分层 Action，跳过旧式 fcurves 计数')
    hide_source_safely(src)
    print('[完成] 只做隐藏收尾；未执行 ARP、rest 校准或脚距补偿。')
    print('[查看] 跳到第1帧以后播放目标模型。Reference.001 若仍显示，请在 Outliner 手动隐藏。')
    print('[保存] 确认动作后另存为新文件，保留重定向前的模板。')


if __name__ == '__main__':
    main()
