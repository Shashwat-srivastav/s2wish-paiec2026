"""Canonical project paths. Every module resolves files through here — no fixed roots."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"                      # measurement-db tables, one folder per benchmark
EMBEDDINGS = DATA / "embed"               # measurement-db-embed vectors
JEV_CACHE = DATA / "jev_cache"            # TypeSafe Jev item scores (offline)

DOCS = ROOT / "docs"
SPEC = DOCS / "spec"
REPORTS = DOCS / "research_notes"

EXPERIMENTS = ROOT / "experiments"        # registry + decision log
RUNS = EXPERIMENTS / "runs"               # per-phase logs, tables, raw results
DECISIONS_LOG = EXPERIMENTS / "decisions.jsonl"

ARTIFACTS = ROOT / "artifacts"            # fitted priors, episode cache
EPISODE_CACHE = ARTIFACTS / "episode_cache"

SUBMISSIONS = ROOT / "submissions"
DIST = SUBMISSIONS / "dist"               # built zips

EXTERNAL = ROOT / "external"              # official, unmodified third-party code
STARTING_KIT = EXTERNAL / "starting_kit"
KIT_TOOLS = STARTING_KIT / "tools"
BASELINE = EXTERNAL / "official_baselines"   # upstream: github.com/aims-foundations/paiec_baseline
MEASUREMENT_DB = EXTERNAL / "measurement-db"
