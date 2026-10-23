"""Clean-clone reproducibility check: clone HEAD, build a fresh venv, link the gated data, run the tests and
scripts/reproduce_all.py --quick, and write a dated report.

    python scripts/clean_clone_check.py                 # full check (fresh venv; needs PyPI access once)
    python scripts/clean_clone_check.py --no-venv       # offline: reuse this interpreter, PYTHONPATH=<clone>/src
    python scripts/clean_clone_check.py --no-overlay    # strictly the committed HEAD (see "overlay" below)

What it does
 1. `git clone` this repository (committed HEAD only) into .tmp/clean_clone/repo.
 2. Data without copying 2.4 GB: src/paiec/paths.py fixes DATA = <repo>/data and has no override, so the clone's
    data/ is a directory junction (Windows) or symlink to this checkout's data/ (or --data DIR).
 3. external/official_baselines (gitignored nested clone, needed by tests/test_simulator.py) is cloned from the local
    copy and checked out at the commit pinned in external/PINS.txt (offline; same commit as upstream).
 4. A fresh venv in .tmp/clean_clone/venv: pip install -r requirements-lock.txt, then pip install --no-deps -e .
    The pip cache and TEMP are redirected to .tmp/ so nothing is written to C:.
 5. pytest -q, then scripts/reproduce_all.py --quick, each through this checkout's scripts/run_slot.py (the
    machine-wide compute slots), one at a time.
 6. Report: experiments/runs/reproducibility/clean_clone_<UTC date-time>.md and .json in THIS checkout.

Overlay: files that are new or modified relative to HEAD can be copied into the clone (default: the
reproducibility files listed in OVERLAY) so they can be exercised before they are committed. The report lists every
overlaid file; once they are committed the overlay is a no-op. Use --no-overlay to test HEAD exactly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ORIGIN = Path(__file__).resolve().parents[1]
WORK = ORIGIN / ".tmp" / "clean_clone"
CLONE = WORK / "repo"
VENV = WORK / "venv"
REPORTS = ORIGIN / "experiments" / "runs" / "reproducibility"
OVERLAY = ["scripts/reproduce_all.py", "scripts/clean_clone_check.py", "tests/test_v1_3_archive.py",
           "scripts/build_v1_3_archive.py", "scripts/table3_official_baseline.py",
           "experiments/runs/review_round1/table3_baseline_seeds0-9.json",
           "requirements-lock.txt", "REPRODUCE.md", "README.md"]


def sh(cmd, cwd=None, env=None, timeout=None, log=None):
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    dtm = time.time() - t0
    if log is not None:
        log.append(dict(cmd=" ".join(str(c) for c in cmd), rc=p.returncode, seconds=round(dtm, 1),
                        tail=(p.stdout + p.stderr)[-3000:]))
    return p, dtm


def rmtree(p: Path):
    if not p.exists():
        return
    link = CLONE / "data"  # the data link: remove the link itself first, never what it points to
    if os.path.lexists(link):
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True)  # no /s: unlinks the junction
        else:
            link.unlink()
    if os.path.lexists(link):
        raise SystemExit(f"could not remove the data link {link}; remove it by hand (rmdir, not del /s)")
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(p)], capture_output=True)
    else:
        shutil.rmtree(p)


def link_dir(link: Path, target: Path) -> str:
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        return "junction"
    link.symlink_to(target, target_is_directory=True)
    return "symlink"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-venv", action="store_true", help="reuse the current interpreter (offline)")
    ap.add_argument("--no-overlay", action="store_true", help="test committed HEAD exactly")
    ap.add_argument("--data", default=str(ORIGIN / "data"), help="existing data/ directory to link")
    ap.add_argument("--skip-pytest", action="store_true")
    a = ap.parse_args()

    t_start = time.time()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    log: list[dict] = []
    rep: dict = dict(started_utc=stamp, origin=str(ORIGIN), clone=str(CLONE), steps=log)
    tmpdir = ORIGIN / ".tmp" / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ, TEMP=str(tmpdir), TMP=str(tmpdir), PIP_CACHE_DIR=str(ORIGIN / ".tmp" / "pip_cache"),
                    PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1")

    # 1. clone
    rmtree(WORK)
    WORK.mkdir(parents=True)
    head, _ = sh(["git", "-C", ORIGIN, "rev-parse", "HEAD"])
    rep["head"] = head.stdout.strip()
    p, _ = sh(["git", "clone", "--quiet", "--no-hardlinks", ORIGIN, CLONE], log=log)
    if p.returncode != 0:
        raise SystemExit("git clone failed:\n" + p.stderr)
    dirty, _ = sh(["git", "-C", ORIGIN, "status", "--porcelain"])
    rep["origin_uncommitted"] = [ln for ln in dirty.stdout.splitlines() if ln.strip()]

    # overlay
    overlaid = []
    if not a.no_overlay:
        for rel in OVERLAY:
            src, dst = ORIGIN / rel, CLONE / rel
            if src.exists() and (not dst.exists() or dst.read_bytes() != src.read_bytes()):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                overlaid.append(rel)
    rep["overlaid_uncommitted_files"] = overlaid
    rep["missing_without_overlay"] = [r for r in OVERLAY if not (CLONE / r).exists() or r in overlaid]

    # 2. data link
    data = Path(a.data).resolve()
    kind = link_dir(CLONE / "data", data)
    rev = (data / "REVISION").read_text().strip() if (data / "REVISION").exists() else "?"
    rep["data"] = dict(target=str(data), link=kind, revision=rev)

    # 3. pinned official baselines
    pins = {ln.split()[0]: ln.split() for ln in (CLONE / "external" / "PINS.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}
    url, commit = pins["official_baselines"][1], pins["official_baselines"][2]
    local = ORIGIN / "external" / "official_baselines"
    src = local if (local / ".git").exists() else url
    sh(["git", "clone", "--quiet", src, CLONE / "external" / "official_baselines"], log=log)
    p, _ = sh(["git", "-C", CLONE / "external" / "official_baselines", "checkout", "--quiet", commit], log=log)
    rep["official_baselines"] = dict(source=str(src), pinned=commit, checkout_rc=p.returncode)

    # 4. environment
    if a.no_venv:
        py = Path(sys.executable)
        env = dict(base_env, PYTHONPATH=str(CLONE / "src"))
        rep["env"] = dict(mode="current interpreter + PYTHONPATH", python=str(py))
    else:
        p, d1 = sh([sys.executable, "-m", "venv", VENV], env=base_env, log=log)
        py = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        d2 = 0.0
        for attempt in range(3):  # PyPI read timeouts happen; the wheel cache in .tmp/pip_cache keeps progress
            p2, dd = sh([py, "-m", "pip", "install", "-q", "--timeout", "120", "--retries", "10", "-r",
                         CLONE / "requirements-lock.txt"], env=base_env, log=log, timeout=3600)
            d2 += dd
            if p2.returncode == 0:
                break
        p3, d3 = sh([py, "-m", "pip", "install", "-q", "--no-deps", "-e", CLONE], env=base_env, log=log, timeout=1800)
        env = dict(base_env)
        env.pop("PYTHONPATH", None)
        frz, _ = sh([py, "-m", "pip", "freeze"], env=env)
        rep["env"] = dict(mode="fresh venv", python=str(py), venv_rc=p.returncode, lock_install_rc=p2.returncode,
                          editable_install_rc=p3.returncode, install_seconds=round(d1 + d2 + d3, 1),
                          freeze=frz.stdout.splitlines())
        if p2.returncode != 0 or p3.returncode != 0:
            rep["env"]["error"] = (p2.stderr + p3.stderr)[-2000:]
            print("environment install failed; rerun, or use --no-venv (offline)", file=sys.stderr)
    ver, _ = sh([py, "-c", "import sys,numpy,scipy,pandas;print(sys.version.split()[0],numpy.__version__,"
                           "scipy.__version__,pandas.__version__)"], env=env)
    rep["env"]["versions"] = ver.stdout.strip() or ver.stderr[-300:]

    slot = [sys.executable, ORIGIN / "scripts" / "run_slot.py", "--"]
    # 5a. pytest
    if not a.skip_pytest:
        p, d = sh(slot + [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={WORK / 'pytest'}"],
                  cwd=CLONE, env=env, log=log, timeout=7200)
        last = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:] or [p.stderr[-200:]]
        rep["pytest"] = dict(rc=p.returncode, seconds=round(d, 1), summary=last[0])
    # 5b. reproduce_all --quick
    script = CLONE / "scripts" / "reproduce_all.py"
    if script.exists():
        out_json = WORK / "reproduce_all_quick.json"
        p, d = sh(slot + [py, script, "--quick", "--json", out_json], cwd=CLONE, env=env, log=log, timeout=7200)
        res = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
        rep["reproduce_all"] = dict(rc=p.returncode, seconds=round(d, 1), counts=res.get("counts"),
                                    not_reproduced=[f"{r['status']} {r['id']}: {r['what']} (got {r['got']}, "
                                                    f"expected {r['expected']}) {r['note']}".strip()
                                                    for r in res.get("rows", [])
                                                    if r["status"] in ("FAIL", "NOSRC", "SKIP")],
                                    warnings=res.get("warnings"))
    else:
        rep["reproduce_all"] = dict(rc=None, note="scripts/reproduce_all.py is not in the clone (not committed)")
    rep["total_seconds"] = round(time.time() - t_start, 1)

    # 6. report
    REPORTS.mkdir(parents=True, exist_ok=True)
    jpath = REPORTS / f"clean_clone_{stamp}.json"
    jpath.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    e = rep["env"]
    lines = [f"# Clean-clone reproducibility check, {stamp}", "",
             f"- HEAD: `{rep['head']}`; clone: `{CLONE}`",
             f"- Uncommitted files overlaid into the clone: {', '.join(f'`{x}`' for x in overlaid) or 'none'}",
             f"- Origin had {len(rep['origin_uncommitted'])} uncommitted paths (not in the clone unless overlaid).",
             f"- Data: {kind} to `{data}` (revision `{rev}`), nothing copied.",
             f"- official_baselines: pinned `{commit[:10]}` checkout rc {rep['official_baselines']['checkout_rc']}",
             f"- Environment: {e['mode']}; versions (python numpy scipy pandas): `{e.get('versions')}`"]
    if "lock_install_rc" in e:
        lines.append(f"- pip install -r requirements-lock.txt rc {e['lock_install_rc']}, editable install rc "
                     f"{e['editable_install_rc']} ({e['install_seconds']} s)")
    if "pytest" in rep:
        lines.append(f"- pytest: rc {rep['pytest']['rc']}, {rep['pytest']['seconds']} s: `{rep['pytest']['summary']}`")
    r = rep["reproduce_all"]
    lines.append(f"- reproduce_all --quick: rc {r.get('rc')}, {r.get('seconds')} s, counts {r.get('counts')}"
                 + (f" ({r['note']})" if r.get("note") else ""))
    for x in r.get("not_reproduced") or []:
        lines.append(f"  - {x}")
    lines += ["", f"Total {rep['total_seconds']} s. Full log: `{jpath.name}`."]
    mpath = REPORTS / f"clean_clone_{stamp}.md"
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    ok = (rep.get("pytest", {}).get("rc", 0) == 0) and r.get("rc") == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
