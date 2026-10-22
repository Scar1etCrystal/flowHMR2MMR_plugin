"""Regression tests for fixed Teto hand-bone resolution.

These tests intentionally use tiny fakes so they can run with ordinary Python;
the resolver itself only relies on Blender's pose-bone collection protocol.
"""

import unittest

from mocap_doctor.presets import resolve_mmd_hand_bones


class _Metadata:
    def __init__(self, name_j):
        self.name_j = name_j


class _Bone:
    def __init__(self, name, name_j="", constraints=()):
        self.name = name
        self.mmd_bone = _Metadata(name_j)
        self.constraints = list(constraints)


class _Constraint:
    type = "IK"

    def __init__(self, subtarget):
        self.subtarget = subtarget


class _Bones:
    def __init__(self, *bones):
        self._bones = list(bones)

    def __iter__(self):
        return iter(self._bones)

    def get(self, name):
        return next((bone for bone in self._bones if bone.name == name), None)


class _Pose:
    def __init__(self, *bones):
        self.bones = _Bones(*bones)


class _Armature:
    def __init__(self, *bones):
        self.pose = _Pose(*bones)


class HandBoneResolutionTests(unittest.TestCase):
    def test_duplicate_import_suffix_is_accepted(self):
        armature = _Armature(
            _Bone("手首.L.001", "左手首"),
            _Bone("手IK.L.001", "左手首"),
        )
        wrist, helper = resolve_mmd_hand_bones(armature, "L")
        self.assertEqual(wrist.name, "手首.L.001")
        self.assertEqual(helper.name, "手IK.L.001")

    def test_mmd_name_fallback_is_used(self):
        armature = _Armature(
            _Bone("wrist_imported", "左手首"),
            _Bone("hand_target_imported", "左手IK"),
        )
        wrist, helper = resolve_mmd_hand_bones(armature, "L")
        self.assertEqual(wrist.name, "wrist_imported")
        self.assertEqual(helper.name, "hand_target_imported")

    def test_ambiguous_candidates_fail_closed(self):
        armature = _Armature(
            _Bone("手首.L"),
            _Bone("手首.L.001"),
        )
        with self.assertRaisesRegex(RuntimeError, "多个候选"):
            resolve_mmd_hand_bones(armature, "L")

    def test_elbow_ik_constraint_resolves_nonstandard_helper(self):
        armature = _Armature(
            _Bone("手首.L", "左手首"),
            _Bone("ひじ.L", constraints=[_Constraint("Teto_Left_Hand_Target")]),
            _Bone("Teto_Left_Hand_Target", "左手首"),
        )
        wrist, helper = resolve_mmd_hand_bones(armature, "L")
        self.assertEqual(wrist.name, "手首.L")
        self.assertEqual(helper.name, "Teto_Left_Hand_Target")

    def test_leg_ik_target_is_not_considered_a_hand_helper(self):
        armature = _Armature(
            _Bone("手首.L", "左手首"),
            _Bone("ひじ.L", constraints=[_Constraint("Teto_Left_Hand_Target")]),
            _Bone("ひざ.L", constraints=[_Constraint("足IK.L")]),
            _Bone("Teto_Left_Hand_Target", "左手首"),
            _Bone("足IK.L", "左足ＩＫ"),
        )
        _wrist, helper = resolve_mmd_hand_bones(armature, "L")
        self.assertEqual(helper.name, "Teto_Left_Hand_Target")

    def test_missing_helper_is_an_explicit_non_error_result(self):
        armature = _Armature(_Bone("手首.L", "左手首"))
        wrist, helper = resolve_mmd_hand_bones(armature, "L")
        self.assertEqual(wrist.name, "手首.L")
        self.assertIsNone(helper)


if __name__ == "__main__":
    unittest.main()
