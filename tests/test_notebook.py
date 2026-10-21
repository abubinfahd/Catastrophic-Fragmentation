import ast
import contextlib
import gc
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from unittest.mock import patch

from benchmark import evaluate, extract_pair, pair_metrics
from statistical_analysis import analyze, confidence_interval, holm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT.parent / 'Reviewer_Audit_2026-10-21/Paper_Token.ipynb'


class FakeTokenizer:
    def __init__(self, extra):
        self.extra = extra

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        count = len(text) + self.extra * (3 if re.search(r'[\u0980-\u09ff]', text) else 1)
        return list(range(count))


class NotebookTests(unittest.TestCase):
    def test_benchmark_matches_actual_notebook_class_and_saved_table(self):
        doc = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        source = ''.join(doc['cells'][3]['source'])
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
        examples = [{'x': 'one', 'y': 'আমি'}, {'nested': {'en': 'two words', 'bn': 'বাংলা কথা'}},
                    {'x': 'no Bengali'}, {'x': 'third valid sentence', 'y': 'তৃতীয় বাংলা বাক্য'}]
        namespace = dict(os=os, gc=gc, re=re, pd=pd, load_dataset=lambda *a, **k: iter(examples))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<original notebook class>', 'exec'), namespace)
        cls = namespace[node.name]
        config = json.loads((ROOT / 'config.json').read_text())
        tokenizers = {name: FakeTokenizer(i+1) for i, name in enumerate(config['models'])}
        original = object.__new__(cls)
        original.domains = {name: [spec] for name, spec in config['domains'].items()}
        original.tokenizers = tokenizers
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory), contextlib.redirect_stdout(io.StringIO()):
            original.output_temp_csv = str(Path(directory) / 'notebook_raw.csv')
            loaded = original.execute_evaluation(sample_limit=3)
            old_raw = pd.read_csv(original.output_temp_csv)
            old_ranking = old_raw.groupby('Model').Compute_Disparity.mean().round(2).sort_index()
            original.compile_final_report(loaded)
            expected = pd.read_csv('frontier_2026_final_benchmark.csv')
            with patch('datasets.load_dataset', side_effect=lambda *a, **k: iter(examples)):
                raw, summary, ranking = evaluate(config, tokenizers, 3, Path(directory)/'new', {})
            pd.testing.assert_frame_equal(raw.drop(columns='Pair'), old_raw)
            pd.testing.assert_frame_equal(summary.reset_index(drop=True), expected.reset_index(drop=True))
            np.testing.assert_array_equal(ranking.set_index('Model').loc[old_ranking.index, 'Grand_Multiplier'], old_ranking)
            self.assertEqual(raw.groupby(['Domain', 'Model']).size().tolist(), [3]*15)

    def test_original_extraction_and_python_rounding(self):
        self.assertEqual(extract_pair({'nested': {'x': ' hello ', 'y': ' আমি '}, 'ignored': ['text']}), ('hello', 'আমি'))
        self.assertEqual(pair_metrics(40, 23, 3, 2), (round(40/3, 2), 11.5, round(23/40, 2)))
        self.assertEqual(pair_metrics(0, 3, 1, 1)[2], 0)

    def test_empirical_tests_use_observed_notebook_values(self):
        rows = []
        for domain, shift in [('one', 0), ('two', .7), ('three', 1.2)]:
            for model, extra in [('a', 0), ('b', .2)]:
                for pair in range(1, 13):
                    rows.append(dict(Domain=domain, Model=model, Pair=pair, Compute_Disparity=round(1+pair/10+shift+extra*(pair%3), 2)))
        raw = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            ci, comparisons, domains = analyze(raw, Path(directory))
        wide = raw[raw.Domain == 'one'].pivot(index='Pair', columns='Model', values='Compute_Disparity')
        actual = comparisons[comparisons.Domain == 'one'].iloc[0]
        expected = stats.ttest_rel(wide.a, wide.b)
        self.assertAlmostEqual(actual['T'], expected.statistic)
        self.assertAlmostEqual(actual.T_p, expected.pvalue)
        w = stats.wilcoxon(np.round(wide.a-wide.b, 2), method='asymptotic')
        self.assertAlmostEqual(actual.Wilcoxon_W, w.statistic)
        self.assertAlmostEqual(actual.Wilcoxon_p, w.pvalue)
        expected_ci = stats.t.interval(.95, 11, loc=wide.a.mean(), scale=stats.sem(wide.a))
        observed_ci = ci[(ci.Domain == 'one') & (ci.Model == 'a')].iloc[0]
        self.assertAlmostEqual(observed_ci.CI_Lower, expected_ci[0])
        self.assertEqual(len(domains), 6)
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_missing_pairs_and_duplicate_observations_are_rejected(self):
        raw = pd.DataFrame([dict(Domain='d', Model='a', Pair=i, Compute_Disparity=i) for i in [1, 2]])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                analyze(pd.concat([raw, raw.iloc[:1]]), Path(directory))
            bad = pd.concat([raw, pd.DataFrame([dict(Domain='d', Model='b', Pair=i, Compute_Disparity=2) for i in [1, 3]])])
            with self.assertRaisesRegex(ValueError, 'same pairs'):
                analyze(bad, Path(directory))


if __name__ == '__main__':
    unittest.main()
