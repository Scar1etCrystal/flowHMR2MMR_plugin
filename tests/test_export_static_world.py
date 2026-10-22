"""export._matrix_world_is_static: the guard that lets the root bake skip
1499 per-frame evaluations.  It must answer True only when nothing on the
object chain can vary over time - every doubt answers False."""

import unittest
from types import SimpleNamespace as NS

from mocap_doctor.core import export


def curve(path):
    return NS(data_path=path)


def action(*paths):
    return NS(fcurves=[curve(p) for p in paths])


def anim(act=None, drivers=(), tracks=()):
    return NS(action=act, drivers=list(drivers), nla_tracks=list(tracks))


def obj(parent=None, animation_data=None, constraints=(), parent_type="OBJECT", rigid_body=None):
    return NS(parent=parent, animation_data=animation_data, constraints=list(constraints),
              parent_type=parent_type, rigid_body=rigid_body)


class StaticWorldTests(unittest.TestCase):
    def test_pose_only_action_under_plain_parents_is_static(self):
        root = obj(animation_data=anim(action('["mmd_prop"]')))
        arm = obj(parent=obj(parent=root),
                  animation_data=anim(action('pose.bones["全ての親"].location',
                                             'pose.bones["腕.L"].rotation_quaternion'),
                                      drivers=[curve('pose.bones["x"].rotation_euler')]))
        self.assertTrue(export._matrix_world_is_static(arm))

    def test_object_transform_curve_anywhere_up_the_chain_is_not_static(self):
        for path in ("location", "rotation_euler", "scale", "delta_location",
                     "rotation_quaternion"):
            correction = obj(animation_data=anim(action(path)))
            arm = obj(parent=obj(parent=correction))
            self.assertFalse(export._matrix_world_is_static(arm), path)

    def test_transform_driver_is_not_static(self):
        arm = obj(animation_data=anim(drivers=[curve("rotation_euler")]))
        self.assertFalse(export._matrix_world_is_static(arm))

    def test_nla_strip_with_transform_or_meta_strip_is_not_static(self):
        clip = NS(type="CLIP", action=action("location"))
        arm = obj(animation_data=anim(tracks=[NS(strips=[clip])]))
        self.assertFalse(export._matrix_world_is_static(arm))
        meta = NS(type="META", action=None)
        arm = obj(animation_data=anim(tracks=[NS(strips=[meta])]))
        self.assertFalse(export._matrix_world_is_static(arm))

    def test_constraints_bone_parent_and_physics_are_not_static(self):
        self.assertFalse(export._matrix_world_is_static(obj(constraints=["CHILD_OF"])))
        self.assertFalse(export._matrix_world_is_static(
            obj(parent=obj(), parent_type="BONE")))
        self.assertFalse(export._matrix_world_is_static(
            obj(parent=obj(rigid_body=object()))))

    def test_action_without_legacy_fcurves_is_not_static(self):
        arm = obj(animation_data=anim(act=NS(fcurves=None)))
        self.assertFalse(export._matrix_world_is_static(arm))


if __name__ == "__main__":
    unittest.main()
