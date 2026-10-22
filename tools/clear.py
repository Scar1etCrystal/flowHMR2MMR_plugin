import bpy

# 选中的骨架对象
obj = bpy.context.active_object

if obj and obj.type == "ARMATURE":
    # 1. 复位所有姿态骨骼
    bpy.ops.object.mode_set(mode="POSE")
    for pbone in obj.pose.bones:
        pbone.location = (0, 0, 0)
        pbone.rotation_euler = (0, 0, 0)
        pbone.rotation_quaternion = (1, 0, 0, 0)
        pbone.scale = (1, 1, 1)

    # 2. 清空关键帧与动作关联
    if obj.animation_data:
        # 清除 Action
        obj.animation_data.action = None
        # 清除所有 NLA 轨道
        if obj.animation_data.nla_tracks:
            for track in list(obj.animation_data.nla_tracks):
                obj.animation_data.nla_tracks.remove(track)

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"{obj.name} 的动画数据已彻底清空并归位！")