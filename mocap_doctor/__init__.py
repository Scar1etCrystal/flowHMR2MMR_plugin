"""MoCap Doctor Blender 4.3 Extension entry point."""


_DEFERRED_KEYMAP_ACTIVE = False
_WINDOWS_CONSOLE_CODE_PAGES = None


def _modules():
    # Keep mocap_doctor.core importable by ordinary Python test runners.
    from . import annotation, operators, preferences, project, properties, ui

    return annotation, operators, preferences, project, properties, ui


def _configure_windows_console_utf8():
    """Make Blender's Windows system console decode our UTF-8 reports."""

    global _WINDOWS_CONSOLE_CODE_PAGES
    import sys

    if sys.platform != "win32" or _WINDOWS_CONSOLE_CODE_PAGES is not None:
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        input_code_page = int(kernel32.GetConsoleCP())
        output_code_page = int(kernel32.GetConsoleOutputCP())
        if not input_code_page and not output_code_page:
            return
        _WINDOWS_CONSOLE_CODE_PAGES = (input_code_page, output_code_page)
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except (AttributeError, OSError, TypeError, ValueError):
        _WINDOWS_CONSOLE_CODE_PAGES = None


def _restore_windows_console_code_pages():
    global _WINDOWS_CONSOLE_CODE_PAGES
    if _WINDOWS_CONSOLE_CODE_PAGES is None:
        return
    try:
        import ctypes

        input_code_page, output_code_page = _WINDOWS_CONSOLE_CODE_PAGES
        kernel32 = ctypes.windll.kernel32
        if input_code_page:
            kernel32.SetConsoleCP(int(input_code_page))
        if output_code_page:
            kernel32.SetConsoleOutputCP(int(output_code_page))
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    finally:
        _WINDOWS_CONSOLE_CODE_PAGES = None


def _register_classes(classes):
    import bpy

    registered = []
    try:
        for cls in classes:
            bpy.utils.register_class(cls)
            registered.append(cls)
    except Exception:
        for cls in reversed(registered):
            try:
                bpy.utils.unregister_class(cls)
            except Exception as exc:
                print(f"[MoCap Doctor] class rollback failed for {cls.__name__}: {exc}")
        raise


def _unregister_classes(classes):
    import bpy

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            # Blender 4.3 has no bpy.utils.is_registered_class().  A missing
            # bl_rna here simply means a partial register never reached it.
            pass
        except Exception as exc:
            print(f"[MoCap Doctor] class unregister failed for {cls.__name__}: {exc}")


def _deferred_register_keymaps():
    global _DEFERRED_KEYMAP_ACTIVE
    if not _DEFERRED_KEYMAP_ACTIVE:
        return None
    try:
        _annotation, _operators, preferences, _project, _properties, _ui = _modules()
        if preferences.register_keymaps():
            _DEFERRED_KEYMAP_ACTIVE = False
            return None
    except Exception as exc:
        _DEFERRED_KEYMAP_ACTIVE = False
        print(f"[MoCap Doctor] deferred keymap registration failed: {exc}")
        return None
    return 0.25


def _register_or_defer_keymaps(preferences):
    global _DEFERRED_KEYMAP_ACTIVE
    import bpy

    if preferences.register_keymaps():
        return
    _DEFERRED_KEYMAP_ACTIVE = True
    if not bpy.app.timers.is_registered(_deferred_register_keymaps):
        bpy.app.timers.register(_deferred_register_keymaps, first_interval=0.1)


def _stop_agent_server():
    """Kill the agent socket server during unregister/quit.

    A live daemon socket thread while bpy tears down its allocators is a
    plausible source of the tbbmalloc access-violation on save+quit - stop it
    before anything else tears down."""
    try:
        from .core import agent_bridge
        agent_bridge.stop_server()
        agent_bridge.stop_fixlist_timer()
    except Exception:
        pass


def _cancel_deferred_keymaps():
    global _DEFERRED_KEYMAP_ACTIVE
    import bpy

    _DEFERRED_KEYMAP_ACTIVE = False
    if bpy.app.timers.is_registered(_deferred_register_keymaps):
        try:
            bpy.app.timers.unregister(_deferred_register_keymaps)
        except (RuntimeError, ValueError):
            pass


def register():
    _configure_windows_console_utf8()
    annotation, operators, preferences, _project, properties, ui = _modules()
    from . import planted_indicators
    from .core import receiver

    undo = []
    try:
        properties.register_properties()
        undo.append(properties.unregister_properties)
        project.register_handlers()
        undo.append(project.unregister_handlers)
        # Q8-2：每次打开任何文件都重新执行 MMR 的 UI 文本块，面板才会回来
        receiver.register_handlers()
        undo.append(receiver.unregister_handlers)
        _register_classes(annotation.CLASSES)
        undo.append(lambda: _unregister_classes(annotation.CLASSES))
        annotation.register_properties()
        undo.append(annotation.unregister_properties)
        planted_indicators.register()
        undo.append(planted_indicators.unregister)
        operators.register_operators()
        undo.append(operators.unregister_operators)
        ui.register_ui()
        undo.append(ui.unregister_ui)
        _register_classes(preferences.CLASSES)
        undo.append(lambda: _unregister_classes(preferences.CLASSES))
        _register_or_defer_keymaps(preferences)
        undo.append(preferences.unregister_keymaps)
        # fixlist 定时器：面板列表的重建/迁移必须发生在 draw 之外
        # （draw 里写 ID 会报 "Writing to ID classes in this context..."）
        try:
            from .core import agent_bridge
            agent_bridge.start_fixlist_timer()
        except Exception as exc:
            print(f"[MoCap Doctor] fixlist timer start failed: {exc}")
    except Exception:
        _cancel_deferred_keymaps()
        for cleanup in reversed(undo):
            try:
                cleanup()
            except Exception as exc:
                print(f"[MoCap Doctor] register rollback failed: {exc}")
        _restore_windows_console_code_pages()
        raise


def unregister():
    annotation, operators, preferences, project, properties, ui = _modules()
    from . import planted_indicators
    from .core import receiver

    _cancel_deferred_keymaps()
    cleanup_steps = (
        # agent socket server first - a live TCP daemon thread during Blender
        # teardown is a plausible tbbmalloc access-violation at exit
        _stop_agent_server,
        planted_indicators.unregister,
        operators.cleanup_annotation_sessions,
        preferences.unregister_keymaps,
        lambda: _unregister_classes(preferences.CLASSES),
        ui.unregister_ui,
        operators.unregister_operators,
        annotation.unregister_properties,
        lambda: _unregister_classes(annotation.CLASSES),
        receiver.unregister_handlers,
        project.unregister_handlers,
        properties.unregister_properties,
    )
    for cleanup in cleanup_steps:
        try:
            cleanup()
        except Exception as exc:
            print(f"[MoCap Doctor] unregister cleanup failed: {exc}")
    _restore_windows_console_code_pages()
