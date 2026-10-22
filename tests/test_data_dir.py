"""resolve_data_dir must never trust a foreign-OS (non-absolute) stored path.
Plain python: python3 tests/test_data_dir.py"""
import os
import sys
import tempfile
import types
import importlib.util
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "mocap_doctor", "project.py"), encoding="utf-8").read()
start = src.index("def _usable_data_dir")
end = src.index("def ensure_project_directories")
ns = {"Path": Path, "bpy": types.SimpleNamespace(data=types.SimpleNamespace(filepath=""))}
exec("def project_data_dir(fp):\n    w = Path(fp)\n    return w.parent / '.mocap_doctor' / w.stem\n" + src[start:end], ns)

fails = []
with tempfile.TemporaryDirectory() as tmp:
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        os.makedirs("F:/proj/data1499")             # the stray relative dir
        blend = os.path.join(tmp, "work", "shot.blend")
        ns["bpy"].data.filepath = blend
        st = types.SimpleNamespace(data_directory="F:/proj/data1499", work_filepath="")
        got = ns["resolve_data_dir"](st)
        want = str(Path(tmp) / "work" / ".mocap_doctor" / "shot")
        ok = got == want and st.data_directory == want
        print(("PASS" if ok else "FAIL"), "foreign drive path ignored even if a stray dir exists:", got)
        if not ok:
            fails.append(1)
        real = os.path.join(tmp, "custom_data")
        os.makedirs(real)
        st2 = types.SimpleNamespace(data_directory=real, work_filepath="")
        ok2 = ns["resolve_data_dir"](st2) == real
        print(("PASS" if ok2 else "FAIL"), "valid absolute custom dir kept")
        if not ok2:
            fails.append(2)
    finally:
        os.chdir(cwd)
        # work file: foreign path → the currently open file (and heal the setting)
        st3 = types.SimpleNamespace(data_directory="", work_filepath="F:/proj/shot.blend")
        got = ns["resolve_work_filepath"](st3)
        ok3 = got == blend and st3.work_filepath == blend
        print(("PASS" if ok3 else "FAIL"), "foreign work_filepath → current file:", got)
        if not ok3:
            fails.append(3)
        st4 = types.SimpleNamespace(data_directory="", work_filepath=os.path.join(real, "w.blend"))
        ok4 = ns["resolve_work_filepath"](st4) == os.path.join(real, "w.blend")
        print(("PASS" if ok4 else "FAIL"), "valid absolute work_filepath kept")
        if not ok4:
            fails.append(4)
sys.exit(1 if fails else 0)
