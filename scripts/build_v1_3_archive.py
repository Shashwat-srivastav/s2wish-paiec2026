"""Deterministically rebuild the candidate archive v1_3_named_facets_noguard.zip from submissions/v1_2.

    python scripts/build_v1_3_archive.py                 # build into .tmp/ and verify against submissions/dist
    python scripts/build_v1_3_archive.py --out FILE.zip  # write the rebuilt archive elsewhere

v1_3 = submissions/v1_2 (model.py, labeling.py unchanged) with two parameter changes in v1_2_params.json:
facets true -> "named" (the model tests truthiness, so this still means all item_features fields) and guard 1 -> 0.
The parameters are re-serialised with json.dumps(indent=1) plus a trailing newline, as in the shipped file.

The shipped zip was written by Python's zipfile on Windows. Its metadata is reproduced here: members are written in
the order labeling.py, model.py, v1_2_params.json; every member is stamped 2026-10-08 02:26:08 with create_system 0
and external_attr 0o600 << 16; and compression is DEFLATE at level 9. With zlib 1.3.x the rebuild is byte-identical
(SHA-256 a4b0e841...). Another zlib may emit different deflate bytes, in which case the check falls back to
content identity (every member's bytes are equal), which is what matters for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submissions" / "v1_2"
SHIPPED = ROOT / "submissions" / "dist" / "v1_3_named_facets_noguard.zip"
SHA256 = "a4b0e841d4a344ff207bed2c855342212e95f7d550195b667446ea4e92685b86"
ORDER = ("labeling.py", "model.py", "v1_2_params.json")
STAMP = (2026, 10, 8, 2, 26, 8)


def members() -> dict[str, bytes]:
    params = json.loads((SRC / "v1_2_params.json").read_bytes())
    if params.get("facets") is not True or params.get("guard") != 1:
        raise SystemExit("submissions/v1_2/v1_2_params.json is not the v1_2 source (facets true, guard 1)")
    params["facets"] = "named"
    params["guard"] = 0
    return {"labeling.py": (SRC / "labeling.py").read_bytes(), "model.py": (SRC / "model.py").read_bytes(),
            "v1_2_params.json": (json.dumps(params, indent=1) + "\n").encode("utf-8")}


def build() -> bytes:
    files = members()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ORDER:
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0o600 << 16
            z.writestr(info, files[name], compresslevel=9)
    return buf.getvalue()


def verify(data: bytes) -> dict:
    new, old = zipfile.ZipFile(io.BytesIO(data)), zipfile.ZipFile(SHIPPED)
    content = sorted(new.namelist()) == sorted(old.namelist()) and all(new.read(n) == old.read(n) for n in ORDER)
    sha = hashlib.sha256(data).hexdigest()
    return {"sha256": sha, "byte_identical": sha == SHA256, "content_identical": content,
            "shipped_sha256_ok": hashlib.sha256(SHIPPED.read_bytes()).hexdigest() == SHA256}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / ".tmp" / "rebuild" / "v1_3_named_facets_noguard.zip"))
    a = ap.parse_args()
    data = build()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    r = verify(data)
    print(json.dumps(dict(out=str(out), **r), indent=1))
    return 0 if r["content_identical"] and r["shipped_sha256_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
