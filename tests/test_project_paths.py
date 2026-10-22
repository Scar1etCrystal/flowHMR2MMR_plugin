"""project.py path rules from the 2026-10-04 review (plain python, no Blender).

M17  save_workfile may only write the file that is OPEN: with a copy open it
     saves the copy (the recorded work file is left alone); with a checkpoint /
     recovery copy open it refuses.  Before, it always wrote the recorded
     work_filepath - opening a checkpoint and pressing 上一步/下一步 overwrote
     the real work file.
M14  step records store absolute checkpoint / report paths; once the project
     moved, the same file name under the current data dir is used.

    PYTHONPATH=. python3 tests/test_project_paths.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path, PureWindowsPath

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "..", "mocap_doctor", "project.py"), encoding="utf-8").read()


def _slice(start_marker, end_marker):
    a = SRC.index(start_marker)
    return SRC[a:SRC.index(end_marker, a)]


ns = {"Path": Path, "PureWindowsPath": PureWindowsPath, "os": os,
      "bpy": types.SimpleNamespace(data=types.SimpleNamespace(filepath=""))}
exec("def _safe_name(v):\n    return v\n"
     + _slice("def project_data_dir", "def _usable_data_dir")
     + _slice("def _usable_data_dir", "def atomic_write_json")
     + _slice("def _same_file", "def save_workfile"), ns)

fails = []


def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL"), name, "::", detail)
    if not ok:
        fails.append(name)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    work = tmp / "proj" / "shot_work.blend"
    data = tmp / "proj" / ".mocap_doctor" / "shot_work"
    for sub in ("checkpoints", "reports", "recovery"):
        (data / sub).mkdir(parents=True)
    work.write_bytes(b"BLEND")
    copy = tmp / "elsewhere" / "shot_copy.blend"
    copy.parent.mkdir()
    copy.write_bytes(b"BLEND")
    ckpt = data / "checkpoints" / "0003_tilt.blend"
    ckpt.write_bytes(b"BLEND")
    st = types.SimpleNamespace(work_filepath=str(work), data_directory=str(data))

    # ---- M17
    ns["bpy"].data.filepath = str(work)
    check("M17a work file open → saves the work file",
          ns["workfile_save_target"](st) == str(work))
    ns["bpy"].data.filepath = str(copy)
    got = ns["workfile_save_target"](st)
    check("M17b a copy open → saves the COPY, never the recorded work file",
          got == str(copy) and st.work_filepath == str(work), got)
    ns["bpy"].data.filepath = str(ckpt)
    try:
        ns["workfile_save_target"](st)
        check("M17c a checkpoint open → refuses", False, "no error")
    except RuntimeError as exc:
        check("M17c a checkpoint open → refuses", "检查点" in str(exc), str(exc)[:40])
    rec = data / "recovery" / "0000_before_restore.blend"
    rec.write_bytes(b"BLEND")
    ns["bpy"].data.filepath = str(rec)
    try:
        ns["workfile_save_target"](st)
        check("M17d a recovery copy open → refuses", False, "no error")
    except RuntimeError:
        check("M17d a recovery copy open → refuses", True)
    # foreign (other-OS) work path: still heals to the open file (unchanged rule)
    st2 = types.SimpleNamespace(work_filepath="F:/proj/shot_work.blend", data_directory=str(data))
    ns["bpy"].data.filepath = str(copy)
    check("M17e foreign work path → the open file (self-heal kept)",
          ns["workfile_save_target"](st2) == str(copy))

    # ---- M14
    ns["bpy"].data.filepath = str(work)
    report = data / "reports" / "0000_contacts.json"
    report.write_text("{}", encoding="utf-8")
    for dead in ("F:\\old\\proj\\.mocap_doctor\\shot_work\\reports\\0000_contacts.json",
                 "/mnt/old_disk/proj/.mocap_doctor/shot_work/reports/0000_contacts.json"):
        got = ns["resolve_project_file"](st, dead, "reports")
        check(f"M14a dead report path → data dir ({dead[:12]}…)", got == str(report), got)
    got = ns["resolve_project_file"](st, "F:/old/checkpoints/0003_tilt.blend", "checkpoints")
    check("M14b dead checkpoint path → data dir", got == str(ckpt), got)
    check("M14c a live absolute path is kept",
          ns["resolve_project_file"](st, str(report), "reports") == str(report))
    missing = "F:/old/checkpoints/0099_gone.blend"
    check("M14d missing everywhere → stored text back (caller reports it)",
          ns["resolve_project_file"](st, missing, "checkpoints") == missing)
    check("M14e empty stays empty", ns["resolve_project_file"](st, "", "reports") == "")

print(f"==== {len(fails)} FAIL ====")
sys.exit(1 if fails else 0)
