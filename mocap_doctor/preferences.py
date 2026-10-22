"""Preferences and native Blender keymap integration for MoCap Doctor."""

from __future__ import annotations

import bpy


ADDON_ID = __package__
SHORTCUT_KEYMAP = "NLA Editor"
LEGACY_SHORTCUT_OPERATOR = "mcd.annotation_quick_menu"
SHORTCUT_DEFINITIONS = (
    {
        "operator": "mcd.annotation_mark_in",
        "label": "标记 / 覆盖入点",
        "type": "I",
        "ctrl": True,
        "shift": True,
    },
    {
        "operator": "mcd.annotation_mark_out",
        "label": "标记 / 覆盖出点",
        "type": "O",
        "ctrl": True,
        "shift": True,
    },
)
SHORTCUT_OPERATORS = tuple(item["operator"] for item in SHORTCUT_DEFINITIONS)

_addon_keymaps: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []


def _iter_own_items(keymap, operator_id=None):
    if keymap is None:
        return
    for item in keymap.keymap_items:
        if item.idname in SHORTCUT_OPERATORS and (operator_id is None or item.idname == operator_id):
            yield item


def get_shortcut_keymap_item(context=None, operator_id=None):
    """Return ``(keyconfig, keymap, item)`` for native preference drawing."""

    context = context or bpy.context
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return None, None, None

    # User keyconfig contains customized copies after a user edits an add-on
    # binding.  Prefer it so the displayed and conflict-tested event is exact.
    for keyconfig in (wm.keyconfigs.user, wm.keyconfigs.addon):
        if keyconfig is None:
            continue
        keymap = keyconfig.keymaps.get(SHORTCUT_KEYMAP)
        for item in _iter_own_items(keymap, operator_id):
            return keyconfig, keymap, item

    for keymap, item in _addon_keymaps:
        if operator_id is None or item.idname == operator_id:
            return wm.keyconfigs.addon, keymap, item
    return None, None, None


def get_shortcut_keymap_items(context=None):
    return [
        (definition, *get_shortcut_keymap_item(context, definition["operator"]))
        for definition in SHORTCUT_DEFINITIONS
    ]


def _event_matches(target, candidate) -> bool:
    try:
        return bool(target.compare(candidate))
    except (AttributeError, TypeError, RuntimeError):
        pass

    if target.type != candidate.type:
        return False
    if target.value != candidate.value and "ANY" not in {target.value, candidate.value}:
        return False
    if target.key_modifier != candidate.key_modifier:
        return False
    if target.any or candidate.any:
        return True
    return all(
        getattr(target, name) == getattr(candidate, name)
        for name in ("shift", "ctrl", "alt", "oskey")
    )


def _relevant_keymap(keymap) -> bool:
    name = keymap.name
    return (
        "NLA" in name
        or name
        in {
            "Animation",
            "Animation Channels",
            "Frames",
            "Screen",
            "Window",
        }
    )


def _item_description(item) -> str:
    description = item.name or item.idname
    if item.idname in {"wm.call_menu", "wm.call_menu_pie"}:
        try:
            menu_name = item.properties.name
        except (AttributeError, TypeError):
            menu_name = ""
        if menu_name:
            description = f"{description} ({menu_name})"
    return description


def get_keymap_conflicts(context=None, target=None) -> list[dict]:
    """Scan effective NLA and parent keymaps for the shortcut event.

    Blender permits duplicate bindings.  This function only reports them; it
    never disables or removes another add-on's keymap item.
    """

    context = context or bpy.context
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return []
    if target is None:
        _kc, _km, target = get_shortcut_keymap_item(context)
    if target is None or not target.active:
        return []

    keyconfigs = []
    for keyconfig in (wm.keyconfigs.active, wm.keyconfigs.user, wm.keyconfigs.addon):
        if keyconfig is not None and keyconfig not in keyconfigs:
            keyconfigs.append(keyconfig)

    conflicts: list[dict] = []
    seen: set[tuple] = set()
    for keyconfig in keyconfigs:
        for keymap in keyconfig.keymaps:
            if keymap.is_modal or not _relevant_keymap(keymap):
                continue
            for candidate in keymap.keymap_items:
                if not candidate.active:
                    continue
                # Copies of this add-on's own item in the user/add-on layers
                # are the same binding, not a conflict.
                if candidate.idname == target.idname:
                    continue
                if not _event_matches(target, candidate):
                    continue
                signature = (
                    keymap.name,
                    candidate.idname,
                    _item_description(candidate),
                    candidate.type,
                    candidate.value,
                    candidate.ctrl,
                    candidate.shift,
                    candidate.alt,
                    candidate.oskey,
                    candidate.key_modifier,
                )
                if signature in seen:
                    continue
                seen.add(signature)
                conflicts.append(
                    {
                        "keyconfig": keyconfig.name,
                        "keymap": keymap.name,
                        "operator": candidate.idname,
                        "label": _item_description(candidate),
                    }
                )
    conflicts.sort(key=lambda item: (item["keymap"], item["label"]))
    return conflicts


class MCD_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    def draw(self, context):
        layout = self.layout
        layout.label(text="NLA 区间端点快捷键", icon="NLA")
        layout.label(text="快捷键只在 NLA Editor 的 MoCap Doctor 标注模式中响应。")

        try:
            import rna_keymap_ui
        except ImportError:
            rna_keymap_ui = None

        for definition, keyconfig, keymap, item in get_shortcut_keymap_items(context):
            box = layout.box()
            box.label(text=definition["label"])
            if item is None:
                box.label(text="快捷键尚未注册；请重新启用插件。", icon="ERROR")
                continue
            try:
                if rna_keymap_ui is None:
                    raise ImportError
                rna_keymap_ui.draw_kmi([], keyconfig, keymap, item, box, 0)
            except (ImportError, AttributeError, RuntimeError, TypeError):
                # A compact native-event fallback for unusual builds.  The
                # Blender 4.3 path above includes modifiers and restore UI.
                row = box.row(align=True)
                row.prop(item, "active", text="")
                row.label(text=item.to_string() or "未设置")

            conflicts = get_keymap_conflicts(context, item)
            if not item.active:
                box.label(text="该快捷键已禁用", icon="INFO")
            elif not conflicts:
                box.label(text="当前 NLA 环境未发现快捷键冲突", icon="CHECKMARK")
            else:
                box.label(text=f"发现 {len(conflicts)} 个可能冲突（不会自动修改）", icon="ERROR")
                for conflict in conflicts[:10]:
                    box.label(text=f"{conflict['keymap']} / {conflict['label']}", icon="DOT")
                if len(conflicts) > 10:
                    box.label(text=f"另有 {len(conflicts) - 10} 项未显示")


CLASSES = (MCD_AddonPreferences,)


def register_keymaps() -> bool:
    if {item.idname for _keymap, item in _addon_keymaps} == set(SHORTCUT_OPERATORS):
        return True
    wm = getattr(bpy.context, "window_manager", None)
    keyconfig = wm.keyconfigs.addon if wm is not None else None
    if keyconfig is None:
        return False
    keymap = keyconfig.keymaps.new(name=SHORTCUT_KEYMAP, space_type="NLA_EDITOR", region_type="WINDOW")
    for legacy in [item for item in keymap.keymap_items if item.idname == LEGACY_SHORTCUT_OPERATOR]:
        try:
            keymap.keymap_items.remove(legacy)
        except (ReferenceError, RuntimeError, ValueError):
            pass
    known = {item.idname for _keymap, item in _addon_keymaps}
    for definition in SHORTCUT_DEFINITIONS:
        operator_id = definition["operator"]
        if operator_id in known:
            continue
        item = next(_iter_own_items(keymap, operator_id), None)
        if item is None:
            item = keymap.keymap_items.new(
                operator_id,
                type=definition["type"],
                value="PRESS",
                ctrl=bool(definition.get("ctrl", False)),
                shift=bool(definition.get("shift", False)),
                alt=bool(definition.get("alt", False)),
            )
            item.active = True
        _addon_keymaps.append((keymap, item))
        known.add(operator_id)
    return known == set(SHORTCUT_OPERATORS)


def unregister_keymaps() -> None:
    for keymap, item in reversed(_addon_keymaps):
        try:
            keymap.keymap_items.remove(item)
        except (ReferenceError, RuntimeError, ValueError):
            pass
    _addon_keymaps.clear()


# Symmetric no-op hooks let a package-level registrar treat modules uniformly.
def register_properties() -> None:
    pass


def unregister_properties() -> None:
    pass
