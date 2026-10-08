"""Golden-master regression check for build_2D_dwf.py (and the dwf2d package behind it).

    python tests/regression.py capture   # record tests/baseline/ from the CURRENT code (do this once, on known-good code)
    python tests/regression.py capture integrated_demo   # record ONE case only (stage 6: never the legacy demos)
    python tests/regression.py check     # re-run everything and compare against tests/baseline/  (exit 1 on any difference)
    python tests/regression.py parity [n]       # stage-6 gate: joint_dwf vs cvxpy, n random cases per family
    python tests/regression.py events [n]       # stage-6 gate: refusal loop vs brute force, post-pass checks
    python tests/regression.py stage6 [n]       # stage-6 checks: presets, invariants, scalability (slow: N = 600)

What is compared, for every demo (legacy_demo, physical_demo, integrated_demo):
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
CASES = ("legacy_demo", "physical_demo", "integrated_demo")
LEGACY_CASES = ("legacy_demo", "physical_demo")      # never re-captured for stage-6 work (implementation.md rule 2)

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


def capture(cases=None):
    """Record the baseline.  cases=None: every case AND the API surface; cases=("integrated_demo",): only those cases
    (the API surface and the other cases' baselines are left as they are)."""
    os.makedirs(BASELINE, exist_ok=True)
    for case in (CASES if cases is None else cases):
        if case not in CASES:
            raise SystemExit(f"unknown case {case!r} (known: {CASES})")
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
    if cases is None:
        with open(os.path.join(BASELINE, "api.json"), "w") as f:
            json.dump(api_surface(), f, indent=1, sort_keys=True)
    print(f"baseline written to {BASELINE}")


_SIG = re.compile(r"^(callable|dataclass)\((.*)\)(.*)$", re.S)


def _additive_signature(old: str, new: str) -> str | None:
    """If `new` only appends parameters / dataclass fields WITH defaults to `old` (same return annotation),
    return the appended part, else None.  Example: callable(a, b=1) -> callable(a, b=1, c=None)."""
    mo, mn = _SIG.match(old), _SIG.match(new)
    if not (mo and mn) or mo.group(1) != mn.group(1) or mo.group(3) != mn.group(3):
        return None
    po, pn = mo.group(2), mn.group(2)
    if not pn.startswith(po) or len(pn) == len(po):
        return None
    extra = pn[len(po):]
    if po and not extra.startswith(", "):
        return None
    params = [p for p in re.split(r",\s(?=\w+\s*[:=])", extra.lstrip(", ")) if p]
    has_default = lambda p: "=" in p and "_MISSING_TYPE" not in p     # dataclass fields without default print MISSING
    return extra if params and all(has_default(p) for p in params) else None


def check(allow_additive: bool = True) -> list[str]:
    """Return a list of differences (empty = identical to the baseline).

    allow_additive (default): changes that only ADD something switched off are reported as "additive: ..."
    notes (via ADDITIVE) and are not differences:
      * a new array key whose value is the off value (NaN, i.e. a new dataclass field left at None),
      * a new build_2D_dwf name,
      * a signature / dataclass that only gains trailing parameters / fields with defaults.
    Every array, stdout, stderr and output file that existed before must still be bit-for-bit identical."""
    ADDITIVE.clear()
    diffs = []
    for case in CASES:
        d = os.path.join(BASELINE, case)
        if not os.path.isdir(d):
            diffs.append(f"{case}: no baseline (python tests/regression.py capture {case})")
            continue
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
                    if (allow_additive and k not in a.files and b[k].dtype.kind == "f" and b[k].ndim == 0
                            and np.isnan(b[k])):
                        ADDITIVE.append(f"{case}: new array {k} = None (off)")
                    elif k not in a.files or k not in b.files:
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
            extra = _additive_signature(sig, got[name]) if allow_additive else None
            if extra:
                ADDITIVE.append(f"api: build_2D_dwf.{name} gained {extra.lstrip(', ')}")
            else:
                diffs.append(f"api: build_2D_dwf.{name} changed: {sig}  ->  {got[name]}")
    if allow_additive:
        ADDITIVE.extend(f"api: new name build_2D_dwf.{name}" for name in sorted(set(got) - set(want)))
    return diffs


ADDITIVE: list[str] = []          # additive-only changes found by the last check()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "capture":                        # capture [case ...]: only the named cases (no API re-capture)
        capture(tuple(sys.argv[2:]) or None)
    elif mode == "check":
        problems = check()
        for p in ADDITIVE:
            print("ADDITIVE", p)
        for p in problems:
            print("DIFF", p)
        print(("IDENTICAL to baseline" + (f" ({len(ADDITIVE)} additive-only change(s), all switched off)" if ADDITIVE else ""))
              if not problems else f"{len(problems)} difference(s)")
        sys.exit(1 if problems else 0)
    elif mode == "parity":                       # stage-6 gate: joint_dwf vs cvxpy, 25 random cases per family
        sys.path.insert(0, PROJECT)
        from dwf2d.verification import parity_matrix
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
        res = parity_matrix(n_cases=n)
        above = sum(r["above"] for r in res.values())
        print(f"total: exact {sum(r['exact'] for r in res.values())}, below {sum(r['below'] for r in res.values())}, "
              f"above {above}, infeasible {sum(r['infeasible'] for r in res.values())}")
        sys.exit(1 if above else 0)
    elif mode == "events":                       # stage-6 gate: refusal loop vs brute force, post-pass checks
        sys.path.insert(0, PROJECT)
        from dwf2d.verification import refusal_gate
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        res = refusal_gate(n_cases=n)
        fails = sum(len(r["failures"]) for r in res.values())
        print("EVENT GATE PASSED" if not fails else f"EVENT GATE: {fails} failure(s)")
        sys.exit(1 if fails else 0)
    elif mode == "stage6":                       # stage-6 checks: presets, invariants, scalability (N = 60, 600)
        sys.path.insert(0, PROJECT)
        from dwf2d.verification import stage6_checks
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        res = stage6_checks(n_cases=n)
        print("STAGE-6 CHECKS PASSED" if not res["failures"] else f"STAGE-6 CHECKS: {len(res['failures'])} failure(s)")
        sys.exit(1 if res["failures"] else 0)
    else:
        raise SystemExit("usage: python tests/regression.py [capture [case] | check | parity [n] | events [n] | stage6 [n]]")
