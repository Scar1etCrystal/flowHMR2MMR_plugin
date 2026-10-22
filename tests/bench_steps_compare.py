"""Compare two bench_wizard / bench_export result files step by step.

    python3 tests/bench_steps_compare.py <before.json> <after.json>

Prints per step: median seconds before/after, output digest SAME/DIFF (keyframe
digest; for the export step the .vmd bytes), object digest when present, and
whether the after-run was deterministic.  Exit code 1 if any digest differs.
"""
import json
import sys


def main(a_path, b_path):
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    bad = 0
    for step, rb in b.items():
        if not isinstance(rb, dict) or "median_s" not in rb:
            continue
        ra = a.get(step)
        if ra is None:
            print(f"{step:18s} (not in {a_path})")
            continue
        same = ra.get("digest") == rb.get("digest")
        obj = ""
        if "obj_digest" in rb:
            osame = ra.get("obj_digest") == rb.get("obj_digest")
            obj = f" objs {'SAME' if osame else 'DIFF'}"
            same = same and osame
        bad += not same
        print(f"{step:18s} {ra['median_s']:7.2f}s -> {rb['median_s']:7.2f}s  "
              f"output {'SAME' if ra.get('digest') == rb.get('digest') else 'DIFF'}{obj}  "
              f"deterministic={rb.get('deterministic')}  {str(rb.get('result'))[:30]}")
    print("STEPS " + ("ALL SAME" if not bad else f"{bad} DIFF"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
