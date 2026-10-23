"""Equivalence + invariant tests: official run_streaming client vs simulator fast path.

Run:  python -m pytest -q tests/test_simulator.py   (or python test_simulator.py)
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch


from paiec import paths
from paiec.evaluation import simulator as sim
from paiec.evaluation.metric import BUDGETS, ece
from paiec.evaluation.synth import make_synthetic, write_sample_data

BASE = paths.BASELINE
MEAN_MODEL = BASE / "empirical_mean" / "model.py"
MEAN_BLE_MODEL = BASE / "empirical_mean_ble_acquisition" / "model.py"
MEAN_BLE_LABELING = BASE / "empirical_mean_ble_acquisition" / "labeling.py"
SAMPLE_LABELING = paths.STARTING_KIT / "sample_code_submission" / "labeling.py"
BASELINE_INGESTION = BASE / "tools" / "streaming_ingestion.py"

CROSS_PAIR_MODEL = '''
def predict(input, labeled=None):
    """Uses ALL revealed labels (other pairs too): sensitive to evidence order/scope."""
    labeled = labeled or []
    if not labeled:
        return 0.5
    same = [y for (s, i), y in labeled if s == input[0] and i["benchmark_id"] == input[1]["benchmark_id"]]
    glob = sum(y for _, y in labeled) / len(labeled)
    if not same:
        return glob
    return (sum(same) + 2 * glob) / (len(same) + 2)
'''


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_ble(input, labeled):
    """Deterministic, network-free stand-in for the BLE LLM probability."""
    key = input[1]["item_content"] + "|" + str(len(labeled))
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2 ** 32


def _legacy_hook(input):  # one-argument -> legacy pool ranking; many ties
    return float(len(input[1]["item_content"]) % 3)


class Fixture:
    tmp = tempfile.TemporaryDirectory(prefix="paiec_synth_")
    root = write_sample_data(make_synthetic(seed=7), Path(tmp.name) / "sample_data")
    rows = sim.load_sample_data_dir(root)
    cross_model = Path(tmp.name) / "cross" / "model.py"
    cross_model.parent.mkdir()
    cross_model.write_text(textwrap.dedent(CROSS_PAIR_MODEL))


def _submissions():
    ble_labeling = _load(MEAN_BLE_LABELING, "paiec_test_mean_ble_labeling")
    sample = _load(SAMPLE_LABELING, "paiec_test_sample_labeling")
    return [
        ("mean+ble_acquisition(kw:labeled,context)", MEAN_BLE_MODEL, ble_labeling.acquisition_function),
        ("mean+random(no hook)", MEAN_MODEL, None),
        ("mean+sample_hook(4 positional,prediction)", MEAN_MODEL, sample.acquisition_function),
        ("mean+legacy_rank(1-arg)", MEAN_MODEL, _legacy_hook),
        ("crosspair+sample_hook", Fixture.cross_model, sample.acquisition_function),
    ], ble_labeling


def _compare(tc, a, b, label):
    tc.assertEqual(set(a["per_pair"]), set(b["per_pair"]), label)
    for key in a["per_pair"]:
        pa, pb = a["per_pair"][key], b["per_pair"][key]
        for b_ in BUDGETS:
            tc.assertEqual(pa["brier"][b_], pb["brier"][b_], f"{label} {key} B{b_}")  # exact
            tc.assertEqual(pa["ece"][b_], pb["ece"][b_], f"{label} {key} ECE{b_}")
        tc.assertEqual(pa["labels"], pb["labels"], label)
        tc.assertEqual(pa["acquired_items"], pb["acquired_items"], label)
    tc.assertEqual(a["overall_alc"], b["overall_alc"], label)


class OfficialEquivalence(unittest.TestCase):
    def test_official_client_matches_fast_path(self):
        pairs = sim.prepare_pairs(Fixture.rows, seed=0)
        subs, ble_labeling = _submissions()
        report = []
        with patch.object(ble_labeling, "_ble_prediction", _fake_ble):
            for name, model_path, hook in subs:
                predict = sim.load_model_module(model_path).predict
                official = sim.run_official_client(pairs, model_path, hook, predict=predict)
                direct = sim.run_direct(pairs, predict, hook)
                _compare(self, official, direct, name)
                # The older paiec_baseline client (no 'configure' event) must agree too.
                legacy_client = _load(BASELINE_INGESTION, "paiec_baseline_streaming_ingestion")
                old = sim.run_official_client(pairs, model_path, hook, predict=predict,
                                              send_configure=False, ingestion_module=legacy_client)
                _compare(self, old, direct, name + " [baseline client]")
                self.assertEqual(official["client_stats"]["acquisition_mode"],
                                 {None: "random"}.get(hook, "legacy_rank" if hook is _legacy_hook else "stream"))
                maxdiff = max(abs(official["per_pair"][k]["brier"][b] - direct["per_pair"][k]["brier"][b])
                              for k in direct["per_pair"] for b in BUDGETS)
                curve = [sum(v["brier"][b] for v in direct["per_pair"].values()) / len(direct["per_pair"])
                         for b in BUDGETS]
                report.append(f"{name:45s} ALC={direct['overall_alc']:.6f} max|dB|={maxdiff:.1e} "
                              f"meanB=" + ",".join(f"{c:.4f}" for c in curve)
                              + f" labels@31={sorted({v['labels'][31] for v in direct['per_pair'].values()})}")
        print("\n" + "\n".join(report))


class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = Fixture.rows
        cls.pairs = sim.prepare_pairs(cls.rows, seed=0)
        cls.mean = staticmethod(sim.load_model_module(MEAN_MODEL).predict)

    def test_eligibility_split_and_repeated_trials(self):
        groups = sim.build_pairs(self.rows)
        sizes = {k: len({r["item_id"] for r in v}) for k, v in groups.items()}
        eligible = {k for k, n in sizes.items() if n >= 80}
        self.assertEqual({p.key for p in self.pairs}, eligible)
        self.assertTrue(any(n == 79 for n in sizes.values()))
        self.assertTrue(any(n == 80 for n in sizes.values()))
        for p in self.pairs:
            n = sizes[p.key]
            self.assertEqual(len(p.acq), (n + 1) // 2)
            eval_items = {r["item_id"] for r in p.eval_rows}
            self.assertEqual(len(eval_items), n // 2)
            self.assertFalse(eval_items & {c["item_id"] for c in p.acq})
            n_rows_eval = sum(1 for r in groups[p.key] if r["item_id"] in eval_items)
            self.assertEqual(len(p.eval_rows), n_rows_eval)  # every repeated run scored
        self.assertTrue(any(len(p.eval_rows) > len({r["item_id"] for r in p.eval_rows}) for p in self.pairs))
        all_pairs = sim.prepare_pairs(self.rows, seed=0, ineligible="eval_all")
        self.assertEqual(len(all_pairs), len(groups))

    def test_random_policy_is_selection_sampling_and_nested(self):
        out = sim.run_direct(self.pairs, self.mean, None, "checkpoint")
        for key, v in out["per_pair"].items():
            for b in BUDGETS:
                self.assertEqual(v["labels"][b], min(b, 31))
            for lo, hi in zip(BUDGETS, BUDGETS[1:]):
                self.assertEqual(v["acquired_items"][hi][:len(v["acquired_items"][lo])], v["acquired_items"][lo])

    def test_empirical_mean_brier_independent_recompute(self):
        out = sim.run_direct(self.pairs, self.mean, None, "checkpoint")
        for p in self.pairs:
            labels = {c["item_id"]: c["y"] for c in p.acq}
            v = out["per_pair"][p.key]
            for b in BUDGETS:
                got = [labels[i] for i in v["acquired_items"][b]]
                pred = sum(got) / len(got) if got else 0.5
                ref = sum((pred - r["y"]) ** 2 for r in p.eval_rows) / len(p.eval_rows)
                self.assertAlmostEqual(v["brier"][b], ref, places=12)

    def test_no_forced_acceptance(self):
        never = lambda input, prediction=None, labeled=None, context=None: False  # noqa: E731
        out = sim.run_direct(self.pairs, self.mean, never, "checkpoint")
        for v in out["per_pair"].values():
            self.assertEqual(set(v["labels"].values()), {0})
            self.assertAlmostEqual(v["alc"], v["brier"][0], places=12)

    def test_constant_half(self):
        res = sim.evaluate(self.rows, lambda i, l=None: 0.5, seeds=(0, 1))
        self.assertAlmostEqual(res["overall_alc_mean"], 0.25, places=12)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            sim.run_direct(self.pairs, lambda i, l=None: 1.5, None)
        with self.assertRaises(ValueError):
            sim.run_direct(self.pairs, self.mean, lambda input, prediction, labeled, context: 1)
        with self.assertRaises(ValueError):
            sim.run_official_client(self.pairs, MEAN_MODEL,
                                    lambda input, prediction, labeled, context: 1, predict=self.mean)

    def test_ece_fixed_bins(self):
        # p=0.3 must fall in [0.3,0.4) (np.linspace edge is 0.30000000000000004)
        self.assertAlmostEqual(ece([1, 0], [0.3, 0.35]), abs(0.325 - 0.5), places=12)
        self.assertAlmostEqual(ece([1], [1.0]), 0.0)

    def test_draw_limit(self):
        chosen = sim.draw_groups(sim.build_pairs(self.rows), limit=1000, seed=0)
        sizes = {k: len({r["item_id"] for r in v}) for k, v in sim.build_pairs(self.rows).items()}
        self.assertLessEqual(sum(sizes[k] for k in chosen), 1000)
        self.assertTrue(all(sizes[k] >= 80 for k in chosen))


if __name__ == "__main__":
    unittest.main(verbosity=2)
