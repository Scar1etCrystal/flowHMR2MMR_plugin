"""One-shot CLI client for the in-Blender agent tool server.

    python agent_client.py ping
    python agent_client.py describe '{"target": "contact.L:0"}'
    python agent_client.py fix_ground '{"side":"L","loc_path":"...","frame_range":[100,130],"mode":"lift"}'

Prints the JSON response.  Use --pretty for indented output.
Set MCD_AGENT_PORT env var to override the default port 6207.
"""

import json
import os
import socket
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HOST = "127.0.0.1"
PORT = int(os.environ.get("MCD_AGENT_PORT", "6207"))


def call(tool: str, args: dict | None = None, timeout: float = 120.0) -> dict:
    req = {"id": 1, "tool": tool, "args": args or {}}
    if os.environ.get("MCD_AGENT_TOKEN"):        # 服务端设了 token 才需要（默认关）
        req["token"] = os.environ["MCD_AGENT_TOKEN"]
    with socket.create_connection((HOST, PORT), timeout=timeout) as sock:
        sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        parts = []
        while True:               # list+join：大响应（bake_range）不再 O(n²) 拼接
            chunk = sock.recv(1 << 20)
            if not chunk:
                break
            parts.append(chunk)
            if b"\n" in chunk:
                break
    buf = b"".join(parts)
    return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))


def main() -> int:
    pretty = "--pretty" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--pretty"]
    if not argv:
        print(__doc__)
        return 2
    tool = argv[0]
    args = json.loads(argv[1]) if len(argv) > 1 else {}
    resp = call(tool, args)
    indent = 1 if pretty else None
    print(json.dumps(resp, ensure_ascii=False, indent=indent))
    return 0 if resp.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
