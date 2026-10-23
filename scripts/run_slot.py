"""Run a command inside one of N machine-wide compute slots, so parallel agents don't exhaust RAM.

    python scripts/run_slot.py -- python -m paiec.analysis.some_experiment --flag
    python scripts/run_slot.py --status

Blocks until a slot is free, then runs the command with BLAS/OpenMP threads capped. A slot is an OS byte-range
lock on experiments/runs/.slots/slotN.lock, held by this wrapper for the life of the job; the OS releases it if
the wrapper dies, so there is no PID bookkeeping, stale-slot reclaim, or unlink race. slotN.info records the
owner (wrapper and child PID) for --status only. The slot count is fixed here, not taken from the environment.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLOT_DIR = ROOT / "experiments" / "runs" / ".slots"
N_SLOTS = 2
THREADS = "3"

if os.name == "nt":
    import msvcrt

    def _lock(fh) -> bool:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
else:
    import fcntl

    def _lock(fh) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def _legacy_busy(i: int) -> bool:
    """Respect slotN.json files written by the previous (file-existence) version while its jobs finish."""
    f = SLOT_DIR / f"slot{i}.json"
    if not f.exists():
        return False
    try:
        pid = json.loads(f.read_text(encoding="utf-8"))["pid"]
    except (ValueError, KeyError, OSError):
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def try_claim():
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(N_SLOTS):
        if _legacy_busy(i):
            continue
        fh = open(SLOT_DIR / f"slot{i}.lock", "a+b")
        if _lock(fh):
            return i, fh
        fh.close()
    return None


def write_info(i: int, info: dict):
    tmp = SLOT_DIR / f"slot{i}.info.tmp"
    tmp.write_text(json.dumps(info), encoding="utf-8")
    os.replace(tmp, SLOT_DIR / f"slot{i}.info")


def status():
    for i in range(N_SLOTS):
        busy = False
        try:
            with open(SLOT_DIR / f"slot{i}.lock", "a+b") as fh:
                busy = not _lock(fh)  # probing releases on close
        except OSError:
            busy = True
        busy = busy or _legacy_busy(i)
        info = SLOT_DIR / f"slot{i}.info"
        detail = info.read_text(encoding="utf-8") if busy and info.exists() else ""
        print(f"slot{i}: {'busy ' + detail if busy else 'free'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.status:
        return status()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        sys.exit("usage: run_slot.py -- <command ...>")
    label, waited = " ".join(cmd)[:200], 0
    while (claim := try_claim()) is None:
        if waited % 300 == 0:
            print(f"[run_slot] all {N_SLOTS} slots busy; waiting ({waited}s)", file=sys.stderr, flush=True)
        time.sleep(15)
        waited += 15
    i, fh = claim
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS,
               MKL_NUM_THREADS=THREADS, NUMEXPR_NUM_THREADS=THREADS)
    try:
        proc = subprocess.Popen(cmd, env=env)
        write_info(i, {"wrapper_pid": os.getpid(), "job_pid": proc.pid, "cmd": label,
                       "since": time.strftime("%Y-%m-%dT%H:%M:%S")})
        sys.exit(proc.wait())
    finally:
        fh.close()  # releases the lock; the lock file itself is never deleted


if __name__ == "__main__":
    main()
