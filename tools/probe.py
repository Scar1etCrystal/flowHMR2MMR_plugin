"""Send a probe file's contents to eval_bpy and pretty-print the result.

    python tools/probe.py .blender_test_tmp/probe_scene.py
    python tools/probe.py .blender_test_tmp/probe_scene.py --raw

The probe file must contain a single Python expression (an immediately
invoked lambda is the usual shape) whose value is repr-able.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_client  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw = "--raw" in sys.argv
    if not argv:
        print(__doc__)
        return 2
    expr = Path(argv[0]).read_text(encoding="utf-8").strip()
    resp = agent_client.call("eval_bpy", {"expr": expr}, timeout=300.0)
    if not resp.get("ok"):
        print(json.dumps(resp, ensure_ascii=False, indent=1))
        return 1
    result = resp["data"]["result"]
    if raw:
        print(result)
        return 0
    try:
        parsed = json.loads(result.strip().strip("'\"").encode().decode("unicode_escape")
                            if result.startswith("'") else result)
        print(json.dumps(parsed, ensure_ascii=False, indent=1))
    except Exception:
        try:
            import ast
            print(json.dumps(ast.literal_eval(result), ensure_ascii=False, indent=1))
        except Exception:
            print(result)
    if resp.get("truncated"):
        print("!! TRUNCATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
