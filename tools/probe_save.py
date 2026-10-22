"""Run a probe file through eval_bpy and save the raw repr result to a file.

    python tools/probe_save.py <probe.py> <out.txt>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_client  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    probe, out = sys.argv[1], sys.argv[2]
    expr = Path(probe).read_text(encoding="utf-8").strip()
    resp = agent_client.call("eval_bpy", {"expr": expr}, timeout=300.0)
    if not resp.get("ok"):
        print(resp.get("error"))
        return 1
    text = resp["data"]["result"]
    Path(out).write_text(text, encoding="utf-8")
    print(f"wrote {len(text)} chars to {out} (truncated={resp.get('truncated')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
