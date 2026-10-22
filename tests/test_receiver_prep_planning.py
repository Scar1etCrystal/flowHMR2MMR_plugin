"""Regression tests for MMR receiver-template prep planning.

Mirrors the fake-based style of test_hand_bone_resolution.py so the decision
logic in core/receiver.py can run with ordinary Python.
"""

import unittest

from mocap_doctor.core import receiver


def _bone(name, name_j="", parent=None, constraints=(), has_children=False,
          has_ik_toggle=True, ik_toggle=True):
    return {
        "name": name,
        "name_j": name_j,
        "parent": parent,
        "constraints": [dict(c) for c in constraints],
        "has_children": has_children,
        "has_ik_toggle": has_ik_toggle,
        "ik_toggle": ik_toggle,
    }


class ConstraintPlanningTests(unittest.TestCase):
    def test_native_leg_ik_and_mmd_tools_rig_are_kept(self):
        bones = [
            _bone("ひじ.R", "右ひじ", parent="腕.R", constraints=[
                {"name": "【IK】R", "type": "IK", "subtarget": "手IK.R"},
                {"name": "MMR_复制变换", "type": "COPY_TRANSFORMS",
                 "subtarget": "ORG-forearm.R_parent"},
            ]),
            _bone("ひざ.R", "右ひざ", constraints=[
                {"name": "IK", "type": "IK", "subtarget": "足ＩＫ.R"},
                {"name": "mmd_ik_limit_override", "type": "LIMIT_ROTATION",
                 "subtarget": ""},
                {"name": "MMR_复制变换", "type": "COPY_TRANSFORMS",
                 "subtarget": "ORG-shin.R_parent"},
            ]),
            _bone("足首.R", "右足首", constraints=[
                {"name": "IK", "type": "IK", "subtarget": "つま先ＩＫ.R"},
            ]),
        ]
        plan = receiver.plan_constraints(bones)
        self.assertEqual(
            [(i["bone"], i["name"]) for i in plan["keep"]],
            [("ひざ.R", "IK"), ("ひざ.R", "mmd_ik_limit_override"),
             ("足首.R", "IK")],
        )
        removed = {(i["bone"], i["name"]) for i in plan["remove"]}
        self.assertEqual(removed, {
            ("ひじ.R", "【IK】R"),
            ("ひじ.R", "MMR_复制变换"),
            ("ひざ.R", "MMR_复制变换"),
        })

    def test_deform_bone_rig_is_kept(self):
        """The leg skin sits on 足D / ひざD / 足首D, so their 付与 rig must live.

        Deleting mmd_additional_rotation / mmd_tools_at_dummy leaves the bones
        animating while the leg mesh stays frozen at rest.
        """
        bones = [
            _bone("足D.R", "右足D", constraints=[
                {"name": "mmd_additional_rotation", "type": "TRANSFORM",
                 "subtarget": "_shadow_足D.R"},
            ]),
            _bone("_shadow_足D.R", constraints=[
                {"name": "mmd_tools_at_dummy", "type": "COPY_TRANSFORMS",
                 "subtarget": "_dummy_足D.R"},
            ]),
        ]
        plan = receiver.plan_constraints(bones)
        self.assertEqual(len(plan["remove"]), 0)
        self.assertEqual(
            [(i["bone"], i["name"]) for i in plan["keep"]],
            [("足D.R", "mmd_additional_rotation"),
             ("_shadow_足D.R", "mmd_tools_at_dummy")],
        )

    def test_constraint_without_subtarget_is_removed(self):
        plan = receiver.plan_constraints([
            _bone("ひざ.R", constraints=[{"type": "LIMIT_ROTATION", "subtarget": ""}]),
        ])
        self.assertEqual(plan["keep"], [])
        self.assertEqual(len(plan["remove"]), 1)


class HelperDeletionTests(unittest.TestCase):
    def test_mmr_helper_sharing_name_j_is_deleted(self):
        bones = [
            _bone("手首.R", "右手首"),
            _bone("手IK.R", "右手首"),
        ]
        plan = receiver.plan_helper_bone_deletion(bones)
        self.assertEqual(plan, {"delete": ["手IK.R"], "skipped": []})

    def test_unique_name_j_bones_are_untouched(self):
        bones = [
            _bone("腕.R", "右腕"),
            _bone("ひじ.R", "右ひじ"),
        ]
        plan = receiver.plan_helper_bone_deletion(bones)
        self.assertEqual(plan, {"delete": [], "skipped": []})

    def test_helper_with_children_is_skipped_not_deleted(self):
        bones = [
            _bone("手首.R", "右手首"),
            _bone("手IK.R", "右手首", has_children=True),
        ]
        plan = receiver.plan_helper_bone_deletion(bones)
        self.assertEqual(plan, {"delete": [], "skipped": ["手IK.R"]})

    def test_half_width_ik_without_duplicate_name_j_is_untouched(self):
        plan = receiver.plan_helper_bone_deletion([_bone("手IK.R", "右手首IK")])
        self.assertEqual(plan, {"delete": [], "skipped": []})


class ElbowReparentTests(unittest.TestCase):
    def test_reparented_elbow_is_restored_to_twist_parent(self):
        bones = {
            "ひじ.R": _bone("ひじ.R", "右ひじ", parent="腕.R"),
            "腕.R": _bone("腕.R", "右腕"),
            "腕捩.R": _bone("腕捩.R", "右腕捩", parent="腕.R"),
        }
        self.assertEqual(receiver.plan_elbow_reparent(bones), [("ひじ.R", "腕捩.R")])

    def test_already_correct_elbow_is_untouched(self):
        bones = {
            "ひじ.R": _bone("ひじ.R", "右ひじ", parent="腕捩.R"),
            "腕捩.R": _bone("腕捩.R", "右腕捩"),
        }
        self.assertEqual(receiver.plan_elbow_reparent(bones), [])

    def test_missing_twist_parent_is_untouched(self):
        bones = {"ひじ.R": _bone("ひじ.R", "右ひじ", parent="腕.R")}
        self.assertEqual(receiver.plan_elbow_reparent(bones), [])


class ArmIKToggleTests(unittest.TestCase):
    def test_only_enabled_arm_bones_are_toggled_off(self):
        bones = {
            "ひじ.R": _bone("ひじ.R", "右ひじ", ik_toggle=True),
            "手首.R": _bone("手首.R", "右手首", ik_toggle=False),
            "ひじ.L": _bone("ひじ.L", "左ひじ", has_ik_toggle=False),
        }
        self.assertEqual(receiver.plan_arm_ik_toggle_off(bones), ["ひじ.R"])


class RigDeletionTests(unittest.TestCase):
    def test_rig_armature_and_ui_text_are_matched(self):
        objects = [
            ("arue式重音テトver 2.01_arm", "ARMATURE"),
            ("RIG-MMR-arue式重音テトver 2.01_arm", "ARMATURE"),
            ("arue式重音テトver 2.01_mesh", "MESH"),
        ]
        texts = ["RIG-MMR-arue式重音テトver 2.01_arm_ui.py", "notes.txt"]
        plan = receiver.plan_rig_deletion(objects, texts)
        self.assertEqual(plan["rig_objects"], ["RIG-MMR-arue式重音テトver 2.01_arm"])
        self.assertEqual(plan["rig_texts"], ["RIG-MMR-arue式重音テトver 2.01_arm_ui.py"])


class StaleActionTests(unittest.TestCase):
    def test_only_pose_bone_actions_are_deleted(self):
        actions = [
            {"name": "Documents_bone", "has_pose_bone_fcurves": True},
            {"name": "walk_cycle", "has_pose_bone_fcurves": False},
        ]
        self.assertEqual(receiver.plan_stale_actions(actions), ["Documents_bone"])


if __name__ == "__main__":
    unittest.main()
