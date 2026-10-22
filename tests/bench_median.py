"""Median-of-runs timing table for two groups of bench_baseline.py outputs.

    python3 tests/bench_median.py "logs/bench_orig*.json" "logs/bench_final*.json"

Each group is a glob; per benchmark the median of the per-run medians is shown.
Golden equality is checked separately with bench_compare.py (first file of each group).
"""
import glob
import json
import statistics
import sys


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")
    return files, [json.load(open(f, encoding="utf-8"))["timings"] for f in files]


def main():
    fa, ta = load(sys.argv[1])
    fb, tb = load(sys.argv[2])
    keys = sorted(set().union(*ta, *tb))
    print(f"before: {len(fa)} runs   after: {len(fb)} runs   (median of run medians, ms)")
    print(f"{'benchmark':<40s} {'before':>10s} {'after':>10s} {'speedup':>8s}  spread(before / after)")
    for k in keys:
        va = [t[k]["median_s"] * 1000 for t in ta if k in t]
        vb = [t[k]["median_s"] * 1000 for t in tb if k in t]
        ma = statistics.median(va) if va else None
        mb = statistics.median(vb) if vb else None
        sp = f"{ma / mb:7.2f}x" if (ma and mb) else ""
        sa = f"{min(va):.1f}-{max(va):.1f}" if va else "-"
        sb = f"{min(vb):.1f}-{max(vb):.1f}" if vb else "-"
        print(f"{k:<40s} {ma if ma is not None else float('nan'):10.1f} "
              f"{mb if mb is not None else float('nan'):10.1f} {sp:>8s}  {sa} / {sb}")


if __name__ == "__main__":
    main()
