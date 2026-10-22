"""The arm chain a hand repair interpolates has to be findable by bone name."""

import unittest

from mocap_doctor import presets


class _Bone:
    def __init__(self, name):
        self.name = name


class _Rig:
    """Minimal stand-in for a source armature."""

    def __init__(self, names):
        self.type = "ARMATURE"
        self.data = type("Data", (), {"bones": [_Bone(name) for name in names]})()
        self.pose = type("Pose", (), {"bones": {name: _Bone(name) for name in names}})()


GVHMR_SUFFIXES = (
    "Pelvis", "L_Ankle", "R_Ankle", "L_Foot", "R_Foot", "L_Wrist", "R_Wrist",
    "L_Collar", "L_Shoulder", "L_Elbow", "R_Collar", "R_Shoulder", "R_Elbow",
    "L_Hip", "L_Knee", "R_Hip", "R_Knee",
)


class GvhmrArmChainTests(unittest.TestCase):
    def setUp(self):
        prefix = presets.GVHMR_PREFIX_CANDIDATES[0]
        self.rig = _Rig([f"{prefix}_{suffix}" for suffix in GVHMR_SUFFIXES])
        self.maps = presets.resolve_source_profile(self.rig, presets.SOURCE_PROFILE_GVHMR)

    def test_chain_is_keyed_by_the_rigs_own_bone_name(self):
        hand_bone = self.maps["bones"]["left_hand"]
        self.assertIn(hand_bone, self.maps["arm_chains"])
        self.assertEqual(self.maps["arm_chains"][hand_bone], (hand_bone,))

    def test_both_sides_resolve(self):
        for side in ("left_hand", "right_hand"):
            bone = self.maps["bones"][side]
            self.assertIsNotNone(self.maps["arm_chains"].get(bone), side)

    def test_repair_leaves_the_clean_arm_alone(self):
        """Only the wrist is HaMeR's; shoulder and elbow come from GVHMR."""

        for side in ("left_hand", "right_hand"):
            chain = self.maps["arm_chains"][self.maps["bones"][side]]
            self.assertEqual(len(chain), 1)
            self.assertTrue(chain[0].endswith("_Wrist"))

    def test_every_chain_bone_exists_on_the_rig(self):
        self.assertEqual(presets.profile_is_available(self.maps, self.rig), [])

    def test_missing_wrist_is_reported(self):
        rig = _Rig([n for n in (f"{presets.GVHMR_PREFIX_CANDIDATES[0]}_{s}"
                                for s in GVHMR_SUFFIXES) if not n.endswith("_L_Wrist")])
        maps = presets.resolve_source_profile(rig, presets.SOURCE_PROFILE_GVHMR)
        self.assertIn(f"{presets.GVHMR_PREFIX_CANDIDATES[0]}_L_Wrist",
                      presets.profile_is_available(maps, rig))


class FreemocapArmChainTests(unittest.TestCase):
    def test_legacy_profile_keeps_the_whole_arm(self):
        rig = _Rig(list(presets.SOURCE_BONES.values()) + presets.ARM_CHAINS["L"]
                   + presets.ARM_CHAINS["R"])
        maps = presets.resolve_source_profile(rig, presets.SOURCE_PROFILE_FREEMOCAP)
        chain = maps["arm_chains"][presets.SOURCE_BONES["left_hand"]]
        self.assertEqual(chain, tuple(presets.ARM_CHAINS["L"]))


if __name__ == "__main__":
    unittest.main()
