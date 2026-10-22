"""打 Blender 扩展包：dist/mocap_doctor-<version>-<日期>.zip

目录结构必须是 <pkg>/blender_manifest.toml（顶层带包名目录），
与 1.7.0 包一致。排除 __pycache__/*.pyc（manifest 的 build 规则同款）。
文件名带日期，方便和上一个包区分（2026-10-06 用户要求）。

tests/ 与 tools/ **不进包**：它们本来就在 mocap_doctor/ 之外，
manifest 的 build.paths_exclude_pattern 也把 tests/ 排除掉了。
"""
import hashlib
import re
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(r"F:\mocap_ai_doctor")
PKG = ROOT / "mocap_doctor"
DIST = ROOT / "dist"
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache"}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}
# 必须进包的模块（新面板相关，2026-10-06）：缺一个就说明打包漏了
REQUIRED_ENTRIES = (
    "mocap_doctor/blender_manifest.toml",
    "mocap_doctor/ui.py",
    "mocap_doctor/operators.py",
    "mocap_doctor/properties.py",
    "mocap_doctor/core/official_import.py",
    "mocap_doctor/core/foot_contacts.py",
    "mocap_doctor/core/foot_spacing.py",
    "mocap_doctor/core/hand_spacing.py",
    "mocap_doctor/core/flowhmr_retarget.py",
    "mocap_doctor/presets/bmap/flowhmr_native_mmd_fix_A_foot_ik.bmap",
)

manifest = (PKG / "blender_manifest.toml").read_text(encoding="utf-8")
version = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M).group(1)
today = datetime.now().strftime("%Y%m%d")
out = DIST / f"mocap_doctor-{version}-{today}.zip"

files = []
for p in sorted(PKG.rglob("*")):
    if p.is_dir():
        continue
    rel = p.relative_to(PKG)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        continue
    if p.suffix in EXCLUDE_SUFFIX:
        continue
    files.append((p, rel))

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p, rel in files:
        arc = f"mocap_doctor/{rel.as_posix()}"
        z.write(p, arc)

print(f"打包完成: {out}")
print(f"  版本   : {version}")
print(f"  日期   : {today}")
print(f"  条目数 : {len(files)}")
print(f"  大小   : {out.stat().st_size:,} 字节 ({out.stat().st_size/1024:.0f} KB)")
print(f"  sha256 : {hashlib.sha256(out.read_bytes()).hexdigest()}")

# 自检
z = zipfile.ZipFile(out)
names = z.namelist()
tops = {n.split("/")[0] for n in names}
assert tops == {"mocap_doctor"}, f"顶层目录不对: {tops}"
assert any(n.endswith("blender_manifest.toml") for n in names), "缺 manifest"
assert not any("__pycache__" in n or n.endswith(".pyc") for n in names), "混入缓存"
assert not any("/tests/" in n or n.startswith("mocap_doctor/tests") for n in names), "混入 tests"
bad = z.testzip()
assert bad is None, f"zip 损坏: {bad}"
missing = [n for n in REQUIRED_ENTRIES if n not in names]
assert not missing, f"缺必需模块: {missing}"
inner = z.read("mocap_doctor/blender_manifest.toml").decode("utf-8")
inner_v = re.search(r'^version\s*=\s*"([^"]+)"', inner, re.M).group(1)
assert inner_v == version, f"包内版本 {inner_v} != 文件名版本 {version}"

print("\n自检通过:")
print(f"  顶层目录 = {tops}")
print(f"  包内版本 = {inner_v}")
print(f"  必需模块 = {len(REQUIRED_ENTRIES)}/{len(REQUIRED_ENTRIES)} 全在"
      f"（含 core/official_import.py / core/foot_contacts.py / core/foot_spacing.py / ui.py）")
print(f"  无 tests/ 条目 = True（按 manifest build 规则与现有约定）")
core = sorted(n.split("/")[-1] for n in names if n.startswith("mocap_doctor/core/"))
print(f"  core 模块 ({len(core)}): {', '.join(core)}")
