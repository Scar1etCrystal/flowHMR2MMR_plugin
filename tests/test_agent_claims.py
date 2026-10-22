"""Pure-Python tests for agent_claims (no Blender): python3 tests/test_agent_claims.py"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "agent_claims", os.path.join(HERE, "..", "mocap_doctor", "core", "agent_claims.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

PARENT = {"hand": "forearm", "forearm": "upper_arm", "upper_arm": "shoulder",
          "f1": "hand", "f2": "f1"}


def anc(b):
    out = set()
    while b in PARENT:
        b = PARENT[b]
        out.add(b)
    return out


fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


t = [1000.0]
L = C.LeaseTable(clock=lambda: t[0])
r = L.claim("a1", {"hand"}, (100, 150), ancestors=anc)
check("claim granted", r["granted"] and r["claim_id"])
r2 = L.claim("a2", {"hand"}, (140, 200), ancestors=anc)
check("overlap denied", not r2["granted"] and r2["conflicts"][0]["agent_id"] == "a1"
      and r2["conflicts"][0]["frames"] == [140, 150], r2["conflicts"])
r3 = L.claim("a2", {"hand"}, (151, 200), ancestors=anc)
check("adjacent ok", r3["granted"])
r4 = L.claim("a3", {"forearm"}, (100, 150), ancestors=anc)
check("parent soft warn", r4["granted"] and r4["related"], r4["related"])
r5 = L.claim("a4", {"forearm"}, (100, 150), ancestors=anc, strict=True)
check("strict denies related", not r5["granted"])
r6 = L.claim("a4", {"f2"}, (100, 150), ancestors=anc, check_only=True)
check("check_only no lease", r6["granted"] and r6["claim_id"] is None and
      not any(c["agent_id"] == "a4" for c in L.table()))
check("covering exact", L.covering("a1", {"hand"}, (110, 140)))
check("covering partial false", not L.covering("a1", {"hand"}, (110, 160)))
L.claim("a1", {"hand"}, (151, 160), ancestors=anc)   # a2 holds 151-200 → denied
check("covering stays false (denied)", not L.covering("a1", {"hand"}, (110, 160)))
L.claim("a5", {"hand", "f1"}, (300, 310), ancestors=anc)
L.claim("a5", {"hand"}, (311, 320), ancestors=anc)
check("covering union", L.covering("a5", {"hand"}, (300, 320)) and
      not L.covering("a5", {"f1"}, (300, 320)))
same = L.claim("a1", {"hand"}, (100, 150), ancestors=anc)
check("same scope renews", same.get("renewed") and same["claim_id"] == r["claim_id"])
t[0] += 2000
check("ttl expiry", not any(c["agent_id"] == "a1" for c in L.table()))
r7 = L.claim("a2", {"hand"}, (100, 150), ancestors=anc)
check("claim after expiry", r7["granted"])
check("release all", L.release("a2") and not L.of("a2"))
live = L.claim("a6", {"f1"}, (500, 510), ancestors=anc)
try:
    L.release("a9", live["claim_id"])
    check("release other's claim refused", False)
except PermissionError:
    check("release other's claim refused", True)
ALLc = L.claim("boss", C.ALL, None, ancestors=anc)
check("ALL claim conflicts everyone", not ALLc["granted"] and ALLc["conflicts"])

J = C.WriteJournal()
J.note(5, "a1", {"hand"}, (100, 150), "hold_pose")
J.note(6, "a2", {"hip"}, (100, 150), "hold_pose")
check("unrelated write passes", J.stale_against(4, 6, "a3", [({"f1"}, (0, 99))], anc) is None)
check("same bone stale", J.stale_against(4, 6, "a3", [({"hand"}, (140, 160))], anc) is not None)
check("ancestor change stale", J.stale_against(4, 6, "a3", [({"f2"}, (120, 130))], anc) is not None)
check("descendant change not stale",
      J.stale_against(4, 6, "a3", [({"forearm"}, (120, 130))], anc) is None)
check("own write not stale", J.stale_against(4, 6, "a1", [({"hand"}, (120, 130))], anc) is None)
J.note(7, None, C.ALL, None, "external")
check("external unscoped stale", J.stale_against(6, 7, "a1", [({"x"}, (1, 2))], anc) is not None)
check("future version stale", J.stale_against(9, 7, "a1", [({"x"}, (1, 2))], anc) is not None)
J2 = C.WriteJournal(maxlen=3)
for v in range(1, 10):
    J2.note(v, "z", {"q"}, (1, 2))
check("truncated journal stale", J2.stale_against(2, 9, "a", [({"nope"}, (50, 60))], anc) is not None)
print(f"\n==== {len(fails)} FAIL ====")
sys.exit(1 if fails else 0)
