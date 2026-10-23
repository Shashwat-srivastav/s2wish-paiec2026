"""Deterministic Codabench submission zip builder.

Usage:
    python scripts/build_zip.py submissions/v1_1 [--out-dir submissions/dist] [--no-sums]
    python scripts/build_zip.py --verify submissions/v1_1 submissions/dist/v1_1.zip

Layout (replicates the shipped dist zips): the files of the submission folder
at the zip root, flat, excluding __pycache__/, *.pyc, analysis.py and README.md.
Members are sorted by name, carry a fixed 1980-01-01 timestamp and fixed
permissions, and are DEFLATE-compressed, so rebuilding is byte-identical.
After building, ``<sha256>  <ver>.zip`` is upserted into <out-dir>/SHA256SUMS.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

EXCLUDE_NAMES = {"analysis.py", "README.md"}
EXCLUDE_DIRS = {"__pycache__"}
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o100644


def member_files(src: Path) -> list[Path]:
    out = []
    for p in sorted(src.iterdir(), key=lambda q: q.name):
        if p.is_dir():
            if p.name in EXCLUDE_DIRS:
                continue
            raise ValueError(f"unexpected subdirectory in submission: {p}")
        if p.name in EXCLUDE_NAMES or p.suffix == ".pyc":
            continue
        out.append(p)
    return out


def build_zip(src: Path, out_zip: Path) -> Path:
    files = member_files(src)
    if not any(p.name == "model.py" for p in files):
        raise ValueError(f"{src} has no model.py")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for p in files:
            info = zipfile.ZipInfo(p.name, date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = FILE_MODE << 16
            info.create_system = 3  # unix, so external_attr is honoured
            zf.writestr(info, p.read_bytes(), compresslevel=9)
    tmp.replace(out_zip)
    return out_zip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_sums(sums_file: Path, zip_path: Path) -> str:
    digest = sha256(zip_path)
    lines = []
    if sums_file.exists():
        lines = [ln for ln in sums_file.read_text().splitlines()
                 if ln.strip() and not ln.rstrip().endswith("  " + zip_path.name)]
    lines.append(f"{digest}  {zip_path.name}")
    lines.sort(key=lambda ln: ln.split("  ", 1)[1])
    sums_file.write_text("\n".join(lines) + "\n", newline="\n")
    return digest


def zip_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {i.filename: zf.read(i) for i in zf.infolist()}


def compare_members(a: Path, b: Path) -> list[str]:
    """Return a list of differences between member contents of two zips."""
    ma, mb = zip_members(a), zip_members(b)
    diffs = [f"only in {a.name}: {n}" for n in sorted(set(ma) - set(mb))]
    diffs += [f"only in {b.name}: {n}" for n in sorted(set(mb) - set(ma))]
    diffs += [f"bytes differ: {n}" for n in sorted(set(ma) & set(mb)) if ma[n] != mb[n]]
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="submissions/<ver> folder")
    ap.add_argument("zip", type=Path, nargs="?", help="(with --verify) zip to compare against")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <src>/../dist")
    ap.add_argument("--no-sums", action="store_true", help="do not touch SHA256SUMS")
    ap.add_argument("--verify", action="store_true",
                    help="compare the member files the builder would ship against ZIP")
    a = ap.parse_args(argv)
    src = a.src.resolve()
    ver = src.name

    if a.verify:
        if a.zip is None:
            ap.error("--verify needs a zip path")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            built = build_zip(src, Path(td) / f"{ver}.zip")
            diffs = compare_members(built, a.zip)
        if diffs:
            print("\n".join(diffs))
            return 1
        print(f"OK: {a.zip} members match {src}")
        return 0

    out_dir = (a.out_dir or src.parent / "dist").resolve()
    out_zip = build_zip(src, out_dir / f"{ver}.zip")
    msg = f"built {out_zip}"
    if not a.no_sums:
        msg += f"  sha256={update_sums(out_dir / 'SHA256SUMS', out_zip)}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
