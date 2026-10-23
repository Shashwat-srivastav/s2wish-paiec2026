"""Build a credential-free BLE archive configured for Mistral.

The runtime reads MISTRAL_API_KEY from the environment. No key is accepted
from this script and no credential is written to the archive.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "external" / "official_baselines"
sys.path.insert(0, str(BASE / "tools"))
from ble_packaging import ble_files, write_archive  # noqa: E402

files = ble_files(BASE, public=True, mode="hf", repo_id="aims-foundations/measurement-db")
cfg = json.loads(files["submission_config.json"].decode("utf-8"))
cfg.update({
    "llm": "mistral/mistral-large-latest",
    "max_steps": 5,
    "question_timeout": 240,
    "api_keys": {"MISTRAL_API_KEY": ""},
})
files["submission_config.json"] = (json.dumps(cfg, indent=2) + "\n").encode("utf-8")
output = BASE / "dist" / "ble_mistral_public.zip"
write_archive(files, output)
print(output)
