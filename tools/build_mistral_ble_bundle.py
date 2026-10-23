"""Build a self-contained, credential-free Mistral BLE archive."""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "external" / "official_baselines"
PAYLOAD = BASE / "payload"
DATA = ROOT / "data"
if PAYLOAD.exists() and any(PAYLOAD.iterdir()):
    raise RuntimeError(f"Refusing to overwrite non-empty payload: {PAYLOAD}")

data_out = PAYLOAD / "data"
data_out.mkdir(parents=True)
eligible = []
manifest_files = []
for bench_dir in sorted(DATA.iterdir()):
    if not bench_dir.is_dir() or bench_dir.name in {"embed", "jev_cache"}:
        continue
    tables = [p for p in bench_dir.glob("*.parquet") if p.name in {
        "benchmarks.parquet", "subjects.parquet", "items.parquet", "response.parquet", "traces.parquet"
    }]
    if not tables:
        continue
    dst = data_out / bench_dir.name
    dst.mkdir()
    eligible.append(bench_dir.name)
    for src in tables:
        shutil.copy2(src, dst / src.name)
        manifest_files.append(f"{bench_dir.name}/{src.name}")

revision = (DATA / "REVISION").read_text(encoding="utf-8").strip()
if len(revision) != 40:
    raise RuntimeError("data/REVISION is not a git SHA")
(data_out / "corpus_manifest.json").write_text(json.dumps({
    "db_repo": "aims-foundations/measurement-db",
    "corpus_scope": "paec-public-training",
    "revision": revision,
    "generated_unix": int(time.time()),
    "eligible_dirs": eligible,
    "files": sorted(manifest_files),
}, indent=2) + "\n", encoding="utf-8")

emb_src = DATA / "embed"
emb_out = PAYLOAD / "embeddings" / "items"
emb_out.mkdir(parents=True)
emb_benches = {}
for src in sorted(emb_src.glob("*.parquet")):
    shutil.copy2(src, emb_out / src.name)
    emb_benches[src.stem] = {"n_items": None}
emb_revision = (DATA / "EMBED_REVISION").read_text(encoding="utf-8").strip()
(PAYLOAD / "embeddings" / "manifest.json").write_text(json.dumps({
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "benchmarks": emb_benches,
    "corpus_scope": "paec-public-training",
    "data_revision": revision,
    "embeddings_repo": "aims-foundations/measurement-db-embed",
    "revision": emb_revision,
}, indent=2) + "\n", encoding="utf-8")

sys.path.insert(0, str(BASE / "tools"))
from ble_packaging import ble_files, write_archive  # noqa: E402

files = ble_files(BASE, public=True, mode="bundle")
cfg = json.loads(files["submission_config.json"].decode("utf-8"))
cfg.update({
    "llm": "mistral/mistral-large-latest",
    "max_steps": 5,
    "question_timeout": 240,
    "hf_data_repo": "",
    "api_keys": {"MISTRAL_API_KEY": ""},
})
files["submission_config.json"] = (json.dumps(cfg, indent=2) + "\n").encode("utf-8")
output = BASE / "dist" / "ble_mistral_bundle_public.zip"
write_archive(files, output)
print(output)
