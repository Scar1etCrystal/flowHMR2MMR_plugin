"""简单开心表情（MMD morph / 形变）—— Blender Text Editor 里 Alt+P 运行。

不写死表情名：脚本自己扫描网格上的 shape key（= MMD 的モーフ），
按关键词自动认出「笑」「眼神笑」「眨眼」，找不到会**如实列出来**让你改。

生成什么：
  * 笑      全程挂着一个温和的笑（开头 1 秒淡入，之后轻微呼吸起伏）
  * 眼睛笑  跟着笑一起给一点（没有这个 morph 就跳过）
  * 眨眼    每 2.5~4.5 秒眨一次（约 4 帧，偶尔连眨两下）—— 有眨眼才像活的

用法：
  MODE='apply'  -> 生成；MODE='remove' -> 撤销（恢复你原来的表情动画，如果有）
  MODE='report' -> 只列出这个模型的所有表情名，什么都不改

不动骨架、不动刚体、不动物理。
"""

import bpy
import json
import math
import random

# ----------------------------------------------------------------- CONFIG
MODE = 'apply'            # apply / remove / report
MESH_NAME = ''            # 留空 = 自动挑 shape key 最多的网格
START_FRAME = None        # None = scene.frame_start
END_FRAME = None          # None = scene.frame_end

SMILE_LEVEL = 0.60        # 笑的程度 0~1（0.6 挺明显但不夸张）
SMILE_BREATH = 0.07       # 笑上的呼吸起伏幅度
EYE_SMILE_RATIO = 0.75    # 眼睛笑相对笑的比例
BLINK_MIN_SEC = 2.5       # 眨眼间隔下限
BLINK_MAX_SEC = 4.5       # 眨眼间隔上限
BLINK_FRAMES = 4          # 一次眨眼几帧（30fps 下 4 帧 ≈ 0.13 秒，接近真人）
DOUBLE_BLINK_CHANCE = 0.25
RANDOM_SEED = 20261010
# ------------------------------------------------------------------------

TAG = 'mcd_face_expression'
STASH_PROP = 'mcd_face_stashed_action'
STASH_FAKE = 'mcd_face_stash_fake_user'
MY_ACTION_PREFIX = 'MCD_FaceExpression_'

SMILE_KEYS = ('笑い', 'にっこり', 'スマイル', '笑顔', 'ニコニコ', 'smile', 'smiling')
# 这些名字看着像笑，其实是别的意思，排除掉（作り笑い = 假笑）
SMILE_EXCLUDE = ('作り笑い', 'fake', '苦笑い')
# 「配套的开心 morph」：开心的眉毛 / 眼睛笑，跟着笑一起给一点（没有就跳过）
HAPPY_EXTRA_KEYS = ('眉にこり', '目笑い', 'にっこり目', '目ニコ', 'eye_smile', 'eyesmile')
BLINK_KEYS = ('まばたき', '瞬き', 'blink', 'ウインク', 'wink')


def log(*a):
    print('[face]', *a)


def pick_mesh():
    if MESH_NAME:
        ob = bpy.data.objects.get(MESH_NAME)
        if ob is None:
            raise RuntimeError('找不到网格 ' + MESH_NAME)
        return ob
    best, best_n = None, 0
    for ob in bpy.data.objects:
        if ob.type != 'MESH' or ob.data.shape_keys is None:
            continue
        n = len(ob.data.shape_keys.key_blocks)
        if n > best_n:
            best, best_n = ob, n
    if best is None:
        raise RuntimeError('这个文件里没有任何带形变(shape key)的网格')
    return best


def key_blocks(ob):
    return ob.data.shape_keys.key_blocks if ob.data.shape_keys else []


def find_morph(ob, keywords, exclude=()):
    """按关键词找 morph。

    先找**完全同名**，再退回子串匹配；并在两层都跳过 exclude。
    为什么必须这样：模型里同时有「笑い」和「作り笑い」（假笑），
    只做子串匹配时 '笑い' 会先命中「作り笑い」—— 效果完全是反的。
    """
    blocks = [kb for kb in key_blocks(ob)
              if not any(x.lower() in kb.name.lower() for x in exclude)]
    for kw in keywords:                          # 1) 精确同名优先
        for kb in blocks:
            if kb.name.strip().lower() == kw.lower():
                return kb
    for kw in keywords:                          # 2) 再退回子串
        for kb in blocks:
            if kw.lower() in kb.name.lower():
                return kb
    return None


def list_morphs(ob):
    blocks = [kb.name for kb in key_blocks(ob)]
    log(f'网格 {ob.name!r}：共 {len(blocks)} 个形变')
    for i in range(0, len(blocks), 5):
        log('   ' + ' | '.join(blocks[i:i + 5]))
    return blocks


def stash_existing(ob):
    """把网格上原有的（非本脚本生成的）表情动画挪到一边，返回其名字。

    关键：绝不能把本脚本自己上一轮生成的动作当成"原有动画"——
    否则你原本的表情动画会在第二次 apply 时被覆盖掉，remove 也恢复不回来。
    """
    key = ob.data.shape_keys
    ad = key.animation_data
    old = ad.action if ad else None
    if old is None:
        return None
    if old.name.startswith(MY_ACTION_PREFIX):
        log(f'上一轮自己生成的动作直接丢弃：{old.name}')
        ad.action = None
        return None
    if ob.get(STASH_PROP):                    # 已经存过原动画了，别再覆盖
        ad.action = None
        return ob[STASH_PROP]
    old.use_fake_user = True                  # 保证不会被垃圾回收
    ob[STASH_PROP] = old.name
    if ad:
        ad.action = None
    log(f'原有表情动画已挪到一边：{old.name}（已设 fake user 不会丢）')
    return old.name


def restore_stashed(ob):
    key = ob.data.shape_keys
    name = ob.get(STASH_PROP)
    if not name:
        return False
    act = bpy.data.actions.get(name)
    if act is not None:
        key.animation_data_create().action = act
        log(f'已恢复原来的表情动画：{name}')
    if STASH_PROP in ob:
        del ob[STASH_PROP]
    return True


def reset_used(ob, used):
    for kb in used:
        kb.value = 0.0


def remove(ob):
    key = ob.data.shape_keys
    ad = key.animation_data
    if ad and ad.action and ad.action.name.startswith(MY_ACTION_PREFIX):
        act = ad.action
        ad.action = None
        if act.users == 0:
            bpy.data.actions.remove(act)
            log('删掉生成的表情动画')
    used = [kb for kb in key_blocks(ob)
            if kb.name in (ob.get('mcd_face_used') or '').split('\n') if kb.name]
    for kb in used:
        kb.value = 0.0
    if 'mcd_face_used' in ob:
        del ob['mcd_face_used']
    restore_stashed(ob)
    return {'removed': True}


def blink_frames(a, b, fps, rng):
    """返回 [(start, peak_frame, end), ...] —— 每 2.5~4.5 秒眨一次。"""
    out = []
    f = a + int(rng.uniform(0.6, 1.8) * fps)
    while f < b - BLINK_FRAMES:
        out.append((f, f + 1, f + BLINK_FRAMES))
        f += int(rng.uniform(BLINK_MIN_SEC, BLINK_MAX_SEC) * fps)
        if rng.random() < DOUBLE_BLINK_CHANCE and f + 8 < b:
            out.append((f, f + 1, f + BLINK_FRAMES))   # 连眨第二下
            f += int(rng.uniform(0.6, 1.2) * fps)
    return out


def run():
    scene = bpy.context.scene
    ob = pick_mesh()
    a = scene.frame_start if START_FRAME is None else int(START_FRAME)
    b = scene.frame_end if END_FRAME is None else int(END_FRAME)
    fps = scene.render.fps / scene.render.fps_base
    list_morphs(ob)

    smile = find_morph(ob, SMILE_KEYS, SMILE_EXCLUDE)
    eye = find_morph(ob, HAPPY_EXTRA_KEYS)
    blink = find_morph(ob, BLINK_KEYS)
    log(f'选中：笑={smile.name if smile else "没找到"}  '
        f'开心配套={eye.name if eye else "没找到"}  '
        f'眨眼={blink.name if blink else "没找到"}')
    if smile is None and blink is None:
        log('一个都没认出来 —— 请从上面的列表里挑名字，填进脚本顶部的关键词，'
            '或者在 Blender 里告诉我名字，我来改。什么都不改，直接结束。')
        return {'ok': False, 'reason': 'no morph matched'}

    stash_existing(ob)
    ad = ob.data.shape_keys.animation_data_create()
    keep = ob.get(STASH_PROP)
    for act in list(bpy.data.actions):
        if act.name.startswith(MY_ACTION_PREFIX) and act.name != keep:
            bpy.data.actions.remove(act)
    action = bpy.data.actions.new(MY_ACTION_PREFIX + ob.name)
    ad.action = action

    rng = random.Random(RANDOM_SEED)
    used = []
    curve = {}

    if smile is not None:
        used.append(smile)
        pts = []
        for f in range(a, b + 1, 3):                              # 每 3 帧一个键
            t = (f - a) / max(1.0, fps)                           # 秒
            fade = min(1.0, t / 1.0)                              # 1 秒淡入
            breath = SMILE_BREATH * math.sin(2 * math.pi * t / 5.3)
            pts.append((f, max(0.0, min(1.0, (SMILE_LEVEL + breath) * fade))))
        if pts[-1][0] != b:
            pts.append((b, pts[-1][1]))
        curve[smile.name] = pts
    if eye is not None and smile is not None:
        used.append(eye)
        curve[eye.name] = [(f, v * EYE_SMILE_RATIO) for f, v in curve[smile.name]]
    if blink is not None:
        used.append(blink)
        seq = []
        for f in range(a, b + 1):
            seq.append([f, 0.0])
        for fs, fp, fe in blink_frames(a, b, fps, rng):
            seq[fs - a][1] = 0.35
            seq[fp - a][1] = 1.0
            seq[fe - a][1] = 0.0
        curve[blink.name] = [(f, v) for f, v in seq]

    for name, pts in curve.items():
        kb = ob.data.shape_keys.key_blocks[name]
        for f, v in pts:
            kb.value = v
            kb.keyframe_insert('value', frame=f)
        if kb.id_data.animation_data and kb.id_data.animation_data.action:
            for fc in kb.id_data.animation_data.action.fcurves:
                if name in fc.data_path:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'
    ob['mcd_face_used'] = '\n'.join(kb.name for kb in used)
    ob[TAG] = True
    scene.frame_set(scene.frame_current)     # 让关键帧接管数值

    info = {'ok': True, 'mesh': ob.name, 'frames': [a, b],
            'smile': smile.name if smile else None,
            'happy_extra': eye.name if eye else None,
            'blink': blink.name if blink else None,
            'blinks': len(blink_frames(a, b, fps, random.Random(RANDOM_SEED)))}
    log('生成完成：', json.dumps(info, ensure_ascii=False))
    return info


if __name__ == '__main__':
    if MODE == 'report':
        list_morphs(pick_mesh())
    elif MODE == 'remove':
        print(remove(pick_mesh()))
    else:
        print(run())
