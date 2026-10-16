import ast
import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sync_notebook import extract_reference
from tokenization.benchmark import evaluate
from tokenization.data import extract_pair
from tokenization.reference import load_reference, run_reference, compare_reference


ROOT = Path(__file__).resolve().parents[1]


class CountTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range({"one two three": 7, "hello": 3, "বাংলা কথা": 10,
                           "আমি": 4, "zero": 0}[text]))


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def test_cli_keeps_prior_log_content_and_appends_printed_results(self):
        output = self.work / "logged_reference"
        output.mkdir()
        log = output / "run.log"
        original = b"Previous run: retained diagnostic output\n"
        log.write_bytes(original)
        command = [sys.executable, "-X", "utf8", str(ROOT / "main.py"),
                   "--mode", "reference", "--no-plots", "--sample-sizes", "10000",
                   "--output", str(output)]
        subprocess.run(command, check=True, capture_output=True, cwd=ROOT)
        first = log.read_bytes()
        self.assertTrue(first.startswith(original))
        self.assertIn(b"REFERENCE REPLAY", first)
        self.assertIn(b"Mistral (7B Instruct v0.3)", first)
        subprocess.run(command, check=True, capture_output=True, cwd=ROOT)
        second = log.read_bytes()
        self.assertTrue(second.startswith(first))
        self.assertGreater(len(second), len(first))
        self.assertEqual(second.count(b"Finished reference run"), 2)

    def test_saved_reference_matches_notebook_and_replays_all_150_values(self):
        reference = load_reference()
        notebook = ROOT.parent / "Paper_Token.ipynb"
        if notebook.exists():
            self.assertEqual(reference, extract_reference(notebook))
        output = self.work / "reference"
        with contextlib.redirect_stdout(io.StringIO()):
            run_reference(ROOT / "config.json", output, no_plots=True)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertFalse(manifest["recomputed"])
        self.assertEqual(manifest["comparison"]["status"], "match")
        self.assertEqual(manifest["comparison"]["compared_values"], 150)
        for run in reference["runs"]:
            with (output / f"n_{run['sample_size']}" / "frontier_2026_final_benchmark.csv").open() as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 15)
            for expected, actual in zip(run["summary"], records):
                self.assertEqual(expected["Model"], actual["Model"])
                for metric in ["Avg_EN_Density", "Avg_BN_Density", "Disparity_Multiplier"]:
                    self.assertEqual(expected[metric], float(actual[metric]))
        for image in (ROOT / "reference").glob("fig*_notebook_display.png"):
            self.assertEqual(image.read_bytes(), (output / image.name).read_bytes())

    def test_fresh_metrics_match_the_actual_notebook_engine(self):
        # Run only the notebook class against in-memory fixtures, with no Hub calls.
        source = (ROOT / "reference" / "notebook_engine.txt").read_text(encoding="utf-8")
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef))
        namespace = dict(os=os, gc=gc, re=re, pd=pd,
                         load_dataset=lambda *args, **kwargs: iter([
                             {"translation": {"en": "one two three", "bn": "বাংলা কথা"}},
                             {"src": "hello", "tgt": "আমি"},
                             {"en": "zero", "bn": "আমি"},
                         ]))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook_class", "exec"), namespace)
        cls = namespace["Frontier2026GrandBenchmarker"]
        runner = cls.__new__(cls)
        runner.tokenizers = {"fixture": CountTokenizer()}
        runner.domains = {"domain": [{"path": "fixture", "config": "bn", "split": "train"}]}
        runner.output_temp_csv = str(self.work / "notebook_raw.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            counts = runner.execute_evaluation(sample_limit=3)
        expected = pd.read_csv(runner.output_temp_csv)
        self.assertEqual(counts, {"domain": 3})
        pairs = self.work / "input"
        pairs.mkdir()
        items = [{"en": "one two three", "bn": "বাংলা কথা"}, {"en": "hello", "bn": "আমি"}, {"en": "zero", "bn": "আমি"}]
        (pairs / "domain_1.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
        output = self.work / "fresh"
        output.mkdir()
        config = {"domains": {"domain": runner.domains["domain"][0]}}
        with contextlib.redirect_stdout(io.StringIO()):
            summary, grand, counts = evaluate(config, runner.tokenizers, {"fixture": "fixture"}, [1, 3], output,
                                               None, {"fixture": "a" * 40}, pairs)
        actual = pd.read_csv(output / "raw_metrics.csv")
        for metric in ["EN_Tokens/Word", "BN_Tokens/Word", "Compute_Disparity"]:
            pd.testing.assert_series_equal(expected[metric], actual[metric])
        final = next(row for row in summary if row["Sample_Size"] == 3)
        self.assertAlmostEqual(final["Avg_EN_Density"], expected["EN_Tokens/Word"].mean())
        self.assertAlmostEqual(final["Avg_BN_Density"], expected["BN_Tokens/Word"].mean())
        self.assertAlmostEqual(final["Disparity_Multiplier"], expected["Compute_Disparity"].mean())
        self.assertAlmostEqual(grand[-1]["Grand_Multiplier"], expected["Compute_Disparity"].mean())
        self.assertEqual(counts["domain"]["sha256"], hashlib.sha256((output / "pairs/domain_1.jsonl").read_bytes()).hexdigest())
        replay = self.work / "recomputed"
        replay.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = evaluate(config, runner.tokenizers, {"fixture": "fixture"}, [1, 3], replay,
                                None, {"fixture": "a" * 40}, output / "pairs")
        self.assertEqual(summary, repeated[0])
        self.assertEqual(grand, repeated[1])
        self.assertEqual(counts["domain"]["sha256"], repeated[2]["domain"]["sha256"])
        self.assertEqual(extract_pair({"ignored": ["English", "বাংলা"], "nested": {"x": "hello", "y": "আমি"}}), ("hello", "আমি"))

    def test_incomplete_dataset_fails_and_preserves_manifest(self):
        from tokenization.pipeline import run
        output = self.work / "failed"
        output.mkdir()
        config = json.loads((ROOT / "config.json").read_text())
        loaded = {name: CountTokenizer() for name in config["models"]}
        repositories = {name: paths[0] for name, paths in config["models"].items()}
        with patch("tokenization.pipeline.load_tokenizers", return_value=(loaded, repositories, {})), \
                patch("tokenization.pipeline.evaluate", return_value=([], [], {"domain": {"pairs": 1}})), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Incomplete dataset"):
                run(ROOT / "config.json", output, [2], no_plots=True)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["domains"]["domain"]["pairs"], 1)
        self.assertFalse((output / "frontier_2026_final_benchmark.csv").exists())

    def test_complete_pipeline_writes_notebook_schema_and_correct_capacities(self):
        from tokenization.pipeline import run
        config = json.loads((ROOT / "config.json").read_text())
        pairs = self.work / "input"
        pairs.mkdir()
        for index in range(1, 4):
            (pairs / f"domain_{index}.jsonl").write_text(
                json.dumps({"en": "hello", "bn": "আমি"}, ensure_ascii=False) + "\n"
                + json.dumps({"en": "one two three", "bn": "বাংলা কথা"}, ensure_ascii=False) + "\n", encoding="utf-8")
        output = self.work / "complete"
        output.mkdir()
        loaded = {name: CountTokenizer() for name in config["models"]}
        repositories = {name: paths[0] for name, paths in config["models"].items()}
        with patch("tokenization.pipeline.load_tokenizers", return_value=(loaded, repositories, {})), \
                contextlib.redirect_stdout(io.StringIO()):
            run(ROOT / "config.json", output, [1, 2], pairs, no_plots=True)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["reference_comparison"]["status"], "unavailable")
        table = pd.read_csv(output / "n_2/frontier_2026_final_benchmark.csv")
        self.assertEqual(len(table), 15)
        self.assertEqual(list(table.columns), ["Domain", "Model", "Avg_EN_Density", "Avg_BN_Density", "Disparity_Multiplier"])
        self.assertTrue((table["Disparity_Multiplier"] == 1.38).all())
        grand = pd.read_csv(output / "model_rankings.csv")
        self.assertTrue((grand.loc[grand["Sample_Size"] == 2, "Pairs"] == 6).all())
        self.assertIn("pairs/domain_1.jsonl", manifest["artifacts"])

    def test_comparison_detects_changed_values(self):
        reference = load_reference()
        summary, rankings = [], []
        for run in reference["runs"]:
            summary += [dict(Sample_Size=run["sample_size"], **row) for row in run["summary"]]
            rankings += [dict(Sample_Size=run["sample_size"], **row) for row in run["rankings"]]
        summary[0]["Disparity_Multiplier"] += 0.01
        result = compare_reference(summary, rankings, self.work)
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["different_values"], 1)

    def test_verify_reference_rejects_mismatch_and_retains_fresh_results(self):
        from tokenization.pipeline import run
        reference = load_reference()
        saved = reference["runs"][0]
        size = saved["sample_size"]
        repositories = {name: paths[0] for name, paths in reference["models"].items()}
        loaded = {name: CountTokenizer() for name in repositories}
        summary = [dict(Sample_Size=size, Pairs=size, Repository=repositories[row["Model"]], **row) for row in saved["summary"]]
        rankings = [dict(Sample_Size=size, Pairs=3 * size, Repository=repositories[row["Model"]], **row) for row in saved["rankings"]]
        summary[0]["Disparity_Multiplier"] += 0.01
        counts = {domain: {"pairs": size} for domain in reference["domains"]}
        output = self.work / "mismatch"
        output.mkdir()
        with patch("tokenization.pipeline.load_tokenizers", return_value=(loaded, repositories, {})), \
                patch("tokenization.pipeline.evaluate", return_value=(summary, rankings, counts)), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "does not exactly match"):
                run(ROOT / "config.json", output, [size], no_plots=True, verify_reference=True)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["reference_comparison"]["different_values"], 1)
        self.assertTrue((output / "frontier_2026_final_benchmark.csv").exists())
        self.assertTrue((output / "report.txt").read_text().startswith("Status: failed"))

    def test_llama_uses_exact_revision_and_generic_config_without_rope_changes(self):
        from tokenization.models import load_tokenizers
        from transformers import AutoTokenizer, PreTrainedConfig, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        fixture = self.work / "tokenizer"
        PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
                               unk_token="[UNK]").save_pretrained(fixture)
        raw = dict(model_type="llama4", tokenizer_class="PreTrainedTokenizerFast",
                   text_config={"rope_parameters": {"rope_type": "llama3", "low_freq_factor": 1.0, "high_freq_factor": 1.0}})
        original = copy.deepcopy(raw)
        real_loader = AutoTokenizer.from_pretrained
        seen = {}

        def local_loader(path, **kwargs):
            seen.update(kwargs)
            self.assertIs(type(kwargs["config"]), PreTrainedConfig)
            kwargs.pop("revision")
            return real_loader(fixture, **kwargs)

        repository = "unsloth/Llama-4-Scout-17B-16E-Instruct"
        with patch("tokenization.models.PreTrainedConfig.get_config_dict", return_value=(raw, {})) as config_loader, \
                patch("tokenization.models.AutoTokenizer.from_pretrained", side_effect=local_loader), \
                contextlib.redirect_stdout(io.StringIO()):
            tokenizers, repositories, failures = load_tokenizers({"Llama": [repository]}, None, revisions={repository: "a" * 40})
        self.assertFalse(failures)
        self.assertEqual(repositories, {"Llama": repository})
        self.assertTrue(tokenizers["Llama"].encode("hello"))
        self.assertEqual(raw, original)
        self.assertEqual(seen["revision"], "a" * 40)
        config_loader.assert_called_once_with(repository, token=None, revision="a" * 40)


if __name__ == "__main__":
    unittest.main()
