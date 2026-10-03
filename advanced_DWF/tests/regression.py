"""Golden-master regression check for build_2D_dwf.py (and the dwf2d package behind it).

    python tests/regression.py capture   # record tests/baseline/ from the CURRENT code (do this once, on known-good code)
    python tests/regression.py check     # re-run everything and compare against tests/baseline/  (exit 1 on any difference)

What is compared, for both demos (legacy_demo, physical_demo):
    stdout            exact text the demo prints (diagnose report, KKT dict, calibration summary)
    stderr            warnings, with the "file:line:" location prefix stripped (code moves change line numbers)
    arrays            every field of JointParams / JointResult / PhysicalStorageResult and the check_kkt_joint dict,
                      compared with np.array_equal (bit-for-bit, NaN == NaN)
    output files      SHA-256 of every PNG and CSV the demo writes
plus the API surface of build_2D_dwf: every name it exposed, and the signature of every callable / dataclass fields.

Each demo runs in a fresh subprocess inside a temporary working directory, so nothing in the project is overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                       # advanced_DWF/
BASELINE = os.path.join(HERE, "baseline")
CASES = ("legacy_demo", "physical_demo")

# Runs inside the subprocess: call one demo, then dump every array of its result to an .npz.
_RUNNER = r"""
import dataclasses, sys, numpy as np
sys.path.insert(0, {project!r})
import build_2D_dwf as M
jp, r, ph = getattr(M, {case!r})()
out = {{}}
def put(prefix, obj):
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            put(prefix + f.name + ".", v)
        elif v is None or isinstance(v, (bool, int, float, np.ndarray, list, tuple)):
            out[prefix + f.name] = np.asarray(np.nan if v is None else v)
        else:
            out[prefix + f.name] = np.asarray(str(v))
put("jp.", jp); put("r.", r); put("ph.", ph)
for k, v in M.check_kkt_joint(jp, r).items():
    out["kkt." + k] = np.asarray(v)
np.savez({npz!r}, **out)
"""

_API = r"""
import dataclasses, inspect, json, sys
sys.path.insert(0, {project!r})
import build_2D_dwf as M
api = {{}}
for name in sorted(dir(M)):
    if name.startswith("__"):
        continue
    obj = getattr(M, name)
    if inspect.ismodule(obj):
        api[name] = "module"
    elif dataclasses.is_dataclass(obj) and isinstance(obj, type):
        api[name] = "dataclass(" + ", ".join(f"{{f.name}}: {{f.type}} = {{f.default!r}}" for f in dataclasses.fields(obj)) + ")"
    elif callable(obj):
        try:
            api[name] = "callable" + str(inspect.signature(obj))
        except (TypeError, ValueError):
            api[name] = "callable"
    else:
        api[name] = "value " + repr(obj)
print(json.dumps(api, indent=1, sort_keys=True))
"""

_LOCATION = re.compile(r"^.*?\.py:\d+: ", re.M)       # "C:\...\build_2D_DWF.py:1876: UserWarning..." -> "UserWarning..."
_SOURCE_ECHO = re.compile(r"^  \S.*\n", re.M)         # the source line Python echoes under a warning
_ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")          # "<... object at 0x1A3A...>" differs on every run


def _norm_stderr(s: str) -> str:
    return _SOURCE_ECHO.sub("", _LOCATION.sub("", s))


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_case(case: str, workdir: str) -> dict:
    npz = os.path.join(workdir, "arrays.npz")
    code = _RUNNER.format(project=PROJECT, case=case, npz=npz)
    p = subprocess.run([sys.executable, "-c", code], cwd=workdir, capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"{case} failed:\n{p.stderr}")
    files = {}
    for root, _, names in os.walk(workdir):
        for n in names:
            if n.endswith((".png", ".csv")):
                full = os.path.join(root, n)
                files[os.path.relpath(full, workdir).replace(os.sep, "/")] = _sha(full)
    return {"stdout": p.stdout, "stderr": _norm_stderr(p.stderr), "files": files, "npz": npz}


def api_surface() -> dict:
    p = subprocess.run([sys.executable, "-c", _API.format(project=PROJECT)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return {k: _ADDRESS.sub("", v) for k, v in json.loads(p.stdout).items()}


def capture():
    os.makedirs(BASELINE, exist_ok=True)
    for case in CASES:
        with tempfile.TemporaryDirectory() as wd:
            res = run_case(case, wd)
            d = os.path.join(BASELINE, case)
            os.makedirs(d, exist_ok=True)
            for key in ("stdout", "stderr"):
                with open(os.path.join(d, key + ".txt"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(res[key])
            with open(os.path.join(d, "files.json"), "w") as f:
                json.dump(res["files"], f, indent=1, sort_keys=True)
            with np.load(res["npz"]) as z:
                np.savez(os.path.join(d, "arrays.npz"), **{k: z[k] for k in z.files})
        print(f"captured {case}: {len(res['files'])} files")
    with open(os.path.join(BASELINE, "api.json"), "w") as f:
        json.dump(api_surface(), f, indent=1, sort_keys=True)
    print(f"baseline written to {BASELINE}")


def check() -> list[str]:
    """Return a list of differences (empty = identical to the baseline)."""
    diffs = []
    for case in CASES:
        d = os.path.join(BASELINE, case)
        with tempfile.TemporaryDirectory() as wd:
            res = run_case(case, wd)
            for key in ("stdout", "stderr"):
                with open(os.path.join(d, key + ".txt"), encoding="utf-8") as f:
                    if f.read() != res[key]:
                        diffs.append(f"{case}: {key} differs")
            with open(os.path.join(d, "files.json")) as f:
                want = json.load(f)
            for name in sorted(set(want) | set(res["files"])):
                if want.get(name) != res["files"].get(name):
                    diffs.append(f"{case}: output file {name} differs (or is missing/new)")
            with np.load(os.path.join(d, "arrays.npz")) as a, np.load(res["npz"]) as b:
                for k in sorted(set(a.files) | set(b.files)):
                    if k not in a.files or k not in b.files:
                        diffs.append(f"{case}: array {k} missing on one side")
                    elif a[k].dtype.kind in "fc":
                        if not np.array_equal(a[k], b[k], equal_nan=True):
                            diffs.append(f"{case}: array {k} differs")
                    elif not np.array_equal(a[k], b[k]):
                        diffs.append(f"{case}: array {k} differs")
    with open(os.path.join(BASELINE, "api.json")) as f:
        want = json.load(f)
    got = api_surface()
    for name, sig in want.items():
        if name not in got:
            diffs.append(f"api: build_2D_dwf.{name} no longer exists")
        elif got[name] != sig:
            diffs.append(f"api: build_2D_dwf.{name} changed: {sig}  ->  {got[name]}")
    return diffs


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "capture":
        capture()
    elif mode == "check":
        problems = check()
        for p in problems:
            print("DIFF", p)
        print("IDENTICAL to baseline" if not problems else f"{len(problems)} difference(s)")
        sys.exit(1 if problems else 0)
    else:
        raise SystemExit("usage: python tests/regression.py [capture | check]")
