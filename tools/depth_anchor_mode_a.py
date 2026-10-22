"""深度 Mode A：在确认整段没有明显深度位移时，稳定 RIG root 的世界 Y。

用途：去掉 root 深度通道的慢漂，不使用单目深度模型。
只写：pose.bones["root"].location[1]（世界 Y/相机深度方向）。
不会写物体 location、root Z、torso_root、脚、旋转或重新导入。

用法：
  MODE='apply'  -> 自动静音 manual2，采样当前可见 root Y，写一条连续 correction NLA
  MODE='remove' -> 删除本脚本 correction，并恢复 manual2 原来的静音状态
  MODE='report' -> 只报告当前 root Y 的漂移，不写任何东西

重要：修改人体动画后需要重新 apply。apply 会先移除本脚本上一版 correction，避免叠加。
"""
import bpy
import json
import math

MODE = 'apply'                    # apply / remove / report
ARM_NAME = 'RIG-arue式重音テトver 2.01_arm'
MANUAL_TRACK = 'manual2'
TRACK_NAME = 'agent_depth_anchor_mode_a'
ACTION_NAME = 'agent_depth_anchor_mode_a_action'
TAG = 'mcd_depth_anchor_mode_a'
LOWPASS_FRAMES = 61               # 30fps 下约 2 秒
MAX_CORRECTION_M = 0.15           # 单帧修正限幅，防止异常值拉飞角色
WRITE_EVERY = 1                   # 每帧写，保证曲线连续


def log(*args):
    print('[depth mode A]', *args, flush=True)


def get_arm():
    arm = bpy.data.objects.get(ARM_NAME)
    if arm is None or arm.type != 'ARMATURE':
        # 兼容用户复制/改名：找同时含 RIG 和 arm 的骨架
        cand = [o for o in bpy.data.objects if o.type == 'ARMATURE'
                and o.name.startswith('RIG-')]
        if len(cand) == 1:
            arm = cand[0]
    if arm is None:
        raise RuntimeError('找不到 RIG armature：' + ARM_NAME)
    return arm


def remove_generated(arm):
    """删除本脚本的 track/action，不碰其他 track。

    Blender 的 NlaTrack 不支持 IDProperties；只能用稳定的 track/strip 名识别。
    """
    ad = arm.animation_data
    if ad:
        for t in list(ad.nla_tracks):
            if t.name == TRACK_NAME:
                ad.nla_tracks.remove(t)
    # ACTION_NAME 是本脚本唯一固定动作名；不要对 Action 调用不必要的 t.get。
    act = bpy.data.actions.get(ACTION_NAME)
    if act is not None and act.users == 0:
        bpy.data.actions.remove(act)


def get_manual_state(arm):
    if not arm.animation_data:
        return None
    for t in arm.animation_data.nla_tracks:
        if t.name == MANUAL_TRACK:
            return bool(t.mute)
    return None


def set_manual_muted(arm, value):
    if not arm.animation_data:
        return False
    for t in arm.animation_data.nla_tracks:
        if t.name == MANUAL_TRACK:
            t.mute = bool(value)
            return True
    return False


def moving_average(values, width):
    """边缘复制的居中滑动平均，纯 Python，Blender 不依赖 numpy。"""
    n = len(values)
    if n == 0:
        return []
    w = max(3, int(width))
    if w % 2 == 0:
        w += 1
    h = w // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        # 边缘段仍用实际窗口，不人为制造大端点波动
        out.append(sum(values[lo:hi]) / max(1, hi - lo))
    return out


def sample_root_y(scene, arm, start, end):
    pb = arm.pose.bones.get('root')
    if pb is None:
        raise RuntimeError('RIG 中没有 root 骨')
    values = []
    for f in range(start, end + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        # 关键：root.location.y 是局部静态通道；可见深度必须读 evaluated world head Y。
        values.append(float((arm.matrix_world @ pb.head).y))
    return values


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def add_action(arm, start, corrections):
    # Action / NlaTrack / NlaStrip 在你的 Blender 版本不支持 IDProperties；
    # 依赖固定的 action/track 名称做清理即可。
    action = bpy.data.actions.new(ACTION_NAME)
    path = 'pose.bones["root"].location'
    fc = action.fcurves.new(data_path=path, index=1)
    pts = list(range(start, start + len(corrections), WRITE_EVERY))
    vals = corrections[::WRITE_EVERY]
    fc.keyframe_points.add(len(pts))
    co = []
    for f, v in zip(pts, vals):
        co.extend((f, float(v)))
    fc.keyframe_points.foreach_set('co', co)
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'
    fc.update()

    ad = arm.animation_data_create()
    track = ad.nla_tracks.new()
    track.name = TRACK_NAME
    strip = track.strips.new(TRACK_NAME, start, action)
    strip.blend_type = 'COMBINE'
    strip.extrapolation = 'NOTHING'
    strip.influence = 1.0
    strip.frame_start = start
    strip.frame_end = start + len(corrections) - 1
    strip.action_frame_start = start
    strip.action_frame_end = start + len(corrections) - 1
    strip.blend_in = 0.0
    strip.blend_out = 0.0
    return track, strip, action


def report(scene, arm, start, end):
    old = scene.frame_current
    y = sample_root_y(scene, arm, start, end)
    scene.frame_set(old)
    low = moving_average(y, LOWPASS_FRAMES)
    corr = [clamp(v - l, -MAX_CORRECTION_M, MAX_CORRECTION_M)
            for v, l in zip(y, low)]
    def stats(a):
        return dict(min=round(min(a), 5), max=round(max(a), 5),
                    p2p=round(max(a) - min(a), 5),
                    mean=round(sum(a) / len(a), 5))
    print(json.dumps({'frames': [start, end], 'root_y': stats(y),
                      'lowpass': stats(low), 'would_apply': stats(corr),
                      'manual2_muted': get_manual_state(arm)},
                     ensure_ascii=False, indent=1))


def apply(scene, arm, start, end):
    prior_manual = get_manual_state(arm)
    # remove/rebuild only our layer; keep manual2's state exactly as found
    remove_generated(arm)
    if not set_manual_muted(arm, True):
        log('找不到 manual2；没有可静音的 manual2 轨（继续执行）')
    else:
        log('manual2 已静音，作为本次 Mode A 的基线')

    old = scene.frame_current
    y = sample_root_y(scene, arm, start, end)
    low = moving_average(y, LOWPASS_FRAMES)
    # 全片稳定深度：去慢漂，保留 root 的高频细节；中位数作为稳定基准。
    ref = sorted(y)[len(y) // 2]
    corrections = [clamp(ref - l, -MAX_CORRECTION_M, MAX_CORRECTION_M)
                   for l in low]
    track, strip, action = add_action(arm, start, corrections)
    arm[ TAG + '_prior_manual_mute'] = -1 if prior_manual is None else int(prior_manual)
    arm[TAG + '_settings'] = json.dumps({
        'frames': [start, end], 'lowpass_frames': LOWPASS_FRAMES,
        'reference_y': ref, 'max_correction_m': MAX_CORRECTION_M,
        'manual2_prior_mute': prior_manual,
    }, ensure_ascii=False)
    scene.frame_set(old)
    bpy.context.view_layer.update()
    log(json.dumps({'track': track.name, 'action': action.name,
                    'frames': [start, end], 'reference_y': round(ref, 5),
                    'correction_p2p': round(max(corrections) - min(corrections), 5),
                    'correction_max_abs': round(max(abs(x) for x in corrections), 5),
                    'manual2_mute': True}, ensure_ascii=False))


def remove(scene, arm):
    remove_generated(arm)
    prior = arm.get(TAG + '_prior_manual_mute', -1)
    if prior == -1:
        # 用户要求暂时禁用 manual2；没有历史值时保持静音
        set_manual_muted(arm, True)
    else:
        set_manual_muted(arm, bool(prior))
    for k in (TAG + '_prior_manual_mute', TAG + '_settings'):
        if k in arm:
            del arm[k]
    scene.frame_set(scene.frame_current)
    bpy.context.view_layer.update()
    log('已删除 Mode A correction；manual2 恢复为 apply 前状态')


def main():
    scene = bpy.context.scene
    arm = get_arm()
    start = scene.frame_start
    end = scene.frame_end
    if MODE == 'report':
        report(scene, arm, start, end)
    elif MODE == 'remove':
        remove(scene, arm)
    else:
        apply(scene, arm, start, end)


if __name__ == '__main__':
    main()
