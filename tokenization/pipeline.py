import json
import logging
import os
import random
import hashlib
import platform
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .benchmark import evaluate
from .models import load_tokenizers
from .plots import save_figures
from .reference import SUMMARY_COLUMNS, RANKING_COLUMNS, compare_reference


def run(config_path, output, sample_sizes=None, pairs_dir=None, no_plots=False, verify_reference=False):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sizes = sorted(set(sample_sizes if sample_sizes is not None else config["sample_sizes"]))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Sample sizes must be positive integers.")
    if not config["models"] or not config["domains"]:
        raise ValueError("Configure at least one model and one domain.")
    if config.get("metric_policy") != "notebook":
        raise ValueError("This project reproduces the notebook's round-before-mean metric policy.")
    if any(len(paths) != 1 for paths in config["models"].values()):
        raise ValueError("Configure exactly one repository per model; substitutions are not reproducible.")
    lock_path = config_path.parent / "repositories.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    revisions = {}
    for kind, paths in [("models", [paths[0] for paths in config["models"].values()]),
                        ("datasets", [spec["path"] for spec in config["domains"].values()])]:
        revisions[kind] = {}
        for path in paths:
            record = lock[kind].get(path, {})
            revision = record.get("revision", "")
            if record.get("repository") != path or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"Missing pinned revision for {path}; regenerate repositories.lock.json explicitly.")
            revisions[kind][path] = revision
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"[SEED] Deterministic sampling and tokenization with Seed: {config['seed']}")
    token = os.environ.get("HF_TOKEN") or None
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "sample_sizes": sizes, "status": "running", "mode": "benchmark", "recomputed": True,
                "repository_lock": lock, "runtime": {"python": sys.version, "platform": platform.platform()},
                "versions": {name: version(name) for name in ["transformers", "tokenizers", "datasets", "huggingface-hub", "pandas", "numpy", "matplotlib", "pyarrow"]},
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")},
                "requirements_lock_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "requirements.lock").read_bytes()).hexdigest()}

    def save_manifest():
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    try:
        tokenizers, repositories, failures = load_tokenizers(config["models"], token, config.get("trust_remote_code", False), revisions["models"])
        manifest.update(loaded_models=repositories, model_load_failures=failures)
        save_manifest()
        missing = [name for name in config["models"] if name not in tokenizers]
        if missing:
            raise RuntimeError(
                "Required tokenizers did not load: " + ", ".join(missing)
                + ". Check run.log, HF_TOKEN, and repository access. "
                "All configured models must load before benchmarking."
            )
        summaries, rankings, counts = evaluate(config, tokenizers, repositories, sizes, output, token, revisions["datasets"], pairs_dir)
        manifest["domains"] = counts
        save_manifest()
        incomplete = [name for name, entry in counts.items() if "error" in entry or entry["pairs"] < max(sizes)]
        if incomplete and config.get("require_full_samples", True):
            raise RuntimeError("Incomplete dataset samples: " + ", ".join(incomplete) + ". Inspect run_manifest.json; no complete benchmark is claimed.")
        if not summaries:
            raise RuntimeError("No valid sentence pairs found. Check dataset configurations and access.")
        summary = pd.DataFrame(summaries).round(2).sort_values(["Sample_Size", "Domain", "Disparity_Multiplier", "Model"], ascending=[True, True, False, True])
        ranking = pd.DataFrame(rankings).round(2).sort_values(["Sample_Size", "Grand_Multiplier", "Model"], ascending=[True, False, True])
        summary.to_csv(output / "benchmark_summary.csv", index=False)
        ranking.to_csv(output / "model_rankings.csv", index=False)
        reports = []
        for size in sizes:
            folder = output / f"n_{size}"
            folder.mkdir(exist_ok=True)
            rows = summary[summary["Sample_Size"] == size]
            rows[SUMMARY_COLUMNS].to_csv(folder / "frontier_2026_final_benchmark.csv", index=False, float_format="%.2f")
            overall = ranking[ranking["Sample_Size"] == size]
            overall[RANKING_COLUMNS].to_csv(folder / "model_rankings.csv", index=False, float_format="%.2f")
            report = "\n" + "=" * 60 + "\nDATASET STREAM LOAD MANIFEST (ACTUAL CAPACITIES):\n" + "=" * 60 + "\n"
            for domain in config["domains"]:
                actual = int(rows.loc[rows["Domain"] == domain, "Pairs"].iloc[0]) if domain in set(rows["Domain"]) else 0
                report += f" -> Domain: {domain:<35} | Actually Loaded & Processed: {actual:,} sentence pairs\n"
            report += "=" * 60 + "\n\n" + "=" * 90 + "\n2026 GRAND PENTAGON LANDSCAPE: EMPIRICAL BENCHMARK\n" + "=" * 90 + "\n"
            report += rows[SUMMARY_COLUMNS].to_string(index=False) + "\n\n" + "-" * 90 + "\n2026 GRAND MODEL INEFFICIENCY RANKING (Worst to Best):\n" + "-" * 90 + "\n"
            report += overall[RANKING_COLUMNS].to_string(index=False) + "\n"
            (folder / "report.txt").write_text(report, encoding="utf-8")
            reports.append(f"Requested N={size:,}\n" + report)
            print(report)
        summary.loc[summary["Sample_Size"] == max(sizes), SUMMARY_COLUMNS].to_csv(output / "frontier_2026_final_benchmark.csv", index=False, float_format="%.2f")
        if not no_plots:
            save_figures(summary, ranking, output)
        comparison = compare_reference(summary.to_dict("records"), ranking.to_dict("records"), output)
        manifest["reference_comparison"] = comparison
        partial = len(tokenizers) != len(config["models"]) or any("error" in entry or entry["pairs"] < max(sizes) for entry in counts.values())
        manifest["status"] = ("failed" if verify_reference and comparison["status"] != "match"
                              else "partial" if partial else "complete")
        (output / "report.txt").write_text(
            f"Status: {manifest['status']}\nNotebook comparison: {comparison['status']}\n"
            "Metrics are rounded per pair before arithmetic averaging, matching the notebook.\n"
            "Grand rankings pool all processed pairs; they are not geometric means.\n\n"
            + "\n\n".join(reports),
            encoding="utf-8")
        logging.info("Run status: %s. Outputs saved to %s", manifest["status"], output)
        if comparison["status"] == "mismatch":
            logging.warning("Fresh output differs from the notebook in %s values; inspect reference_comparison.csv", comparison["different_values"])
        if verify_reference and comparison["status"] != "match":
            raise RuntimeError("Fresh output does not exactly match the saved notebook; see reference_comparison.csv.")
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        artifacts = {}
        for path in output.rglob("*"):
            if path.is_file() and path.name not in ("run_manifest.json", "run.log"):
                with path.open("rb") as handle:
                    artifacts[path.relative_to(output).as_posix()] = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest["artifacts"] = artifacts
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
