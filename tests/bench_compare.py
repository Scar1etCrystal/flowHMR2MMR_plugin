"""Compare two bench_baseline.py outputs: exact golden diff + timing table.

    python3 tests/bench_compare.py logs/bench_before.json logs/bench_after.json

Exit 0 = golden outputs identical (任务2 铁律：任何数值输出变化 = 回滚).
Floats are compared exactly (JSON keeps repr precision); any difference is
listed with its key path and |Δ|.
"""
import json
import sys


def walk(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append((f"{path}.{k}", "missing in " + ("before" if k not in a else "after"), None))
                continue
            walk(a[k], b[k], f"{path}.{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"len {len(a)} != {len(b)}", None))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            diffs.append((path, f"{a!r} != {b!r}", None))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b:
            diffs.append((path, f"{a!r} != {b!r}", abs(float(a) - float(b))))
    elif a != b:
        diffs.append((path, f"{str(a)[:80]!r} != {str(b)[:80]!r}", None))


def main():
    before = json.load(open(sys.argv[1], encoding="utf-8"))
    after = json.load(open(sys.argv[2], encoding="utf-8"))
    diffs = []
    # 工具表增减不是数值输出：单独报告，不算 golden 差异
    tb_ = (before["golden"].get("ping") or {}).pop("tools", None)
    ta_ = (after["golden"].get("ping") or {}).pop("tools", None)
    if tb_ is not None and ta_ is not None and tb_ != ta_:
        print(f"INFO tools added={sorted(set(ta_) - set(tb_))} "
              f"removed={sorted(set(tb_) - set(ta_))}")
    walk(before["golden"], after["golden"], "golden", diffs)
    tb, ta = before["timings"], after["timings"]
    print(f"{'benchmark':<40s} {'before ms':>11s} {'after ms':>11s} {'speedup':>8s}")
    for k in sorted(set(tb) | set(ta)):
        b = tb.get(k, {}).get("median_s")
        a = ta.get(k, {}).get("median_s")
        bs = f"{b * 1000:11.1f}" if b is not None else f"{'-':>11s}"
        as_ = f"{a * 1000:11.1f}" if a is not None else f"{'-':>11s}"
        sp = f"{b / a:7.2f}x" if (a and b) else f"{'':>8s}"
        print(f"{k:<40s} {bs} {as_} {sp}")
    print()
    if diffs:
        num = [d for d in diffs if d[2] is not None]
        print(f"GOLDEN DIFF: {len(diffs)} differences "
              f"(numeric {len(num)}, max |Δ|={max((d[2] for d in num), default=0):.3e})")
        sections = sorted({p.split(".")[1].split("[")[0] for p, _m, _d in diffs if p.count(".") >= 1})
        print(f"GOLDEN DIFF SECTIONS: {sections}")
        for p, msg, _ in diffs[:60]:
            print(f"  {p}: {msg}")
        return 1
    print(f"GOLDEN IDENTICAL ({len(before['golden'])} sections, exact float equality)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
