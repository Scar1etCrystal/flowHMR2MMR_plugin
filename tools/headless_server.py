# -*- coding: utf-8 -*-
"""在后台 Blender 里起 agent socket 服务（127.0.0.1:6207，MCD_AGENT_PORT 可覆盖）。

    blender -b work/fixture_1499_3.blend --python tools/headless_server.py

后台模式没有 GUI 事件循环，bpy.app.timers 不会走——本脚本自己循环调
_pump() 顶替定时器。socket 线程收请求进队列，这里在主线程排水执行。

退出 = Ctrl+C / kill。**进程一关内存里的 op 就没了**，所以远程 agent
每完成一段工作必须先调 `save` 工具写盘。
"""
import sys
import time

import bpy  # noqa: F401  （触发扩展侧已注册的模块进 sys.modules）


def _find_bridge():
    for name, mod in sys.modules.items():
        if name == "agent_bridge" or name.endswith(".core.agent_bridge"):
            return mod
    return None


def main():
    ab = _find_bridge()
    if ab is None:
        raise SystemExit(
            "[headless] agent_bridge 没载入——扩展没启用？"
            "先跑一次 blender -b --python tools/enable_addons.py")
    info = ab.start_server()
    scene = bpy.context.scene
    print(f"[headless] server={info} scene={scene.name} "
          f"frames={scene.frame_start}-{scene.frame_end}", flush=True)
    print("[headless] pumping... (kill 本进程即停)", flush=True)
    loop = getattr(ab, "headless_pump_loop", None)
    if loop is not None:          # 事件驱动：请求一到立刻处理（无 70ms 轮询延迟）
        loop()
    while True:                   # 旧扩展回退
        ab._pump()
        time.sleep(ab.TIMER_INTERVAL)


if __name__ == "__main__":
    main()
