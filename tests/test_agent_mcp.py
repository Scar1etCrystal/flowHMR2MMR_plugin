"""tools/agent_mcp.py (review M10): the MCP bridge must forward EVERY tool the
in-Blender server registers - its old static whitelist stopped at 24 tools, so
claim/release/save/eval_bpy and all 13 plugin tools came back "未知工具" over MCP.

Plain python (no Blender): a fake JSON-lines server stands in for agent_bridge.
    PYTHONPATH=. python3 tests/test_agent_mcp.py
"""
import contextlib
import importlib.util
import io
import json
import os
import socket
import socketserver
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_TOOLS = ["ping", "describe", "hold_pose", "motion_copy", "markers",
                "claim", "save", "eval_bpy", "foot_lock"]
SEEN = []


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            req = json.loads(line.decode("utf-8"))
            tool = req.get("tool")
            SEEN.append(tool)
            if tool == "ping":
                resp = {"ok": True, "data": {"tools": SERVER_TOOLS}}
            elif tool in SERVER_TOOLS:
                resp = {"ok": True, "tool": tool, "data": {"echo": req.get("args")}}
            else:
                resp = {"ok": False, "tool": tool,
                        "error": {"code": "E_UNKNOWN", "message": f"未知工具 {tool!r}"}}
            self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
            self.wfile.flush()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_mcp(port=None):
    if port is None:
        os.environ.pop("MCD_AGENT_PORT", None)
    else:
        os.environ["MCD_AGENT_PORT"] = str(port)
    spec = importlib.util.spec_from_file_location(
        "agent_mcp_under_test", os.path.join(HERE, "..", "tools", "agent_mcp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_default_port():
    mod = _load_mcp(None)
    check("default MCP port is the Windows-safe 6207", mod.PORT == 6207, mod.PORT)
    return mod


def _rpc(mod, method, params=None, mid=1):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._handle({"jsonrpc": "2.0", "id": mid, "method": method,
                     "params": params or {}})
    return json.loads(buf.getvalue().strip().splitlines()[-1])


fails = []


def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL"), name, "::", detail)
    if not ok:
        fails.append(name)


_check_default_port()
port = _free_port()
srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), _Handler)
srv.daemon_threads = True
threading.Thread(target=srv.serve_forever, daemon=True).start()
mcp = _load_mcp(port)
try:
    listed = [t["name"] for t in _rpc(mcp, "tools/list")["result"]["tools"]]
    check("tools/list = the server's live tool table (plugin tools included)",
          listed == SERVER_TOOLS, listed)
    descs = {t["name"]: t["description"] for t in _rpc(mcp, "tools/list")["result"]["tools"]}
    check("known tools keep their static description",
          descs["hold_pose"].startswith("通用姿态保持"), descs["hold_pose"][:20])
    r = _rpc(mcp, "tools/call", {"name": "motion_copy", "arguments": {"bones": ["x"]}})
    check("a plugin tool is forwarded, not rejected locally",
          not r["result"]["isError"] and SEEN[-1] == "motion_copy",
          r["result"]["content"][0]["text"][:80])
    r = _rpc(mcp, "tools/call", {"name": "no_such_tool", "arguments": {}})
    check("unknown names go to the server (E_UNKNOWN comes back from there)",
          r["result"]["isError"] and SEEN[-1] == "no_such_tool"
          and "E_UNKNOWN" in r["result"]["content"][0]["text"], SEEN[-1])
finally:
    srv.shutdown()
    srv.server_close()

# server down: tools/list falls back to the static table, calls say "连不上"
mcp_down = _load_mcp(_free_port())
listed = [t["name"] for t in _rpc(mcp_down, "tools/list")["result"]["tools"]]
check("server down → static fallback list", "hold_pose" in listed and "ping" in listed,
      len(listed))
r = _rpc(mcp_down, "tools/call", {"name": "motion_copy", "arguments": {}})
check("server down → a readable connection error",
      r["result"]["isError"] and "连不上" in r["result"]["content"][0]["text"], "")

print(f"==== {len(fails)} FAIL ====")
sys.exit(1 if fails else 0)
