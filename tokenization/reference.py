"""Replay archived notebook outputs. This does not recompute tokenization."""

import csv
import hashlib
import json
import shutil
from pathlib import Path


SUMMARY_COLUMNS = ["Domain", "Model", "Avg_EN_Density", "Avg_BN_Density", "Disparity_Multiplier"]
RANKING_COLUMNS = ["Model", "Grand_Multiplier"]
METRICS = SUMMARY_COLUMNS[2:]


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.2f}" if isinstance(row[key], float) else row[key] for key in columns})


def load_reference():
    path = Path(__file__).resolve().parents[1] / "reference" / "notebook_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reference(summary, rankings, output):
    reference = load_reference()
    observed = {(int(row["Sample_Size"]), row["Domain"], row["Model"]): row for row in summary}
    grand = {(int(row["Sample_Size"]), row["Model"]): row for row in rankings}
    sizes = {int(row["Sample_Size"]) for row in summary}
    differences = []
    for run in reference["runs"]:
        size = run["sample_size"]
        if size not in sizes:
            continue
        for expected in run["summary"]:
            actual = observed.get((size, expected["Domain"], expected["Model"]))
            for metric in METRICS:
                value = float(actual[metric]) if actual is not None else None
                differences.append(dict(Sample_Size=size, Domain=expected["Domain"], Model=expected["Model"],
                                        Metric=metric, Reference=expected[metric], Observed=value,
                                        Match=value == expected[metric]))
        for expected in run["rankings"]:
            actual = grand.get((size, expected["Model"]))
            value = float(actual["Grand_Multiplier"]) if actual is not None else None
            differences.append(dict(Sample_Size=size, Domain="pooled", Model=expected["Model"],
                                    Metric="Grand_Multiplier", Reference=expected["Grand_Multiplier"],
                                    Observed=value, Match=value == expected["Grand_Multiplier"]))
    path = output / "reference_comparison.csv"
    write_csv(path, differences, ["Sample_Size", "Domain", "Model", "Metric", "Reference", "Observed", "Match"])
    unmatched = sum(not row["Match"] for row in differences)
    return {"status": ("match" if unmatched == 0 else "mismatch") if differences else "unavailable",
            "compared_values": len(differences), "different_values": unmatched,
            "notebook_sha256": reference["notebook_sha256"]}


def run_reference(config_path, output, sizes=None, no_plots=False):
    reference = load_reference()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["models"] != reference["models"] or config["domains"] != reference["domains"]
            or config["seed"] != reference["seed"] or config.get("metric_policy") != "notebook"):
        raise ValueError("Reference replay requires the master notebook's models and datasets.")
    sizes = sorted(set(sizes if sizes is not None else config["sample_sizes"]))
    known = {run["sample_size"]: run for run in reference["runs"]}
    if not sizes or set(sizes) - set(known):
        raise ValueError("Reference replay supports only the notebook's recorded 10000, 50000, and 100000 runs.")
    output.mkdir(parents=True, exist_ok=True)
    summaries, rankings, reports = [], [], []
    for size in sizes:
        run = known[size]
        folder = output / f"n_{size}"
        folder.mkdir(exist_ok=True)
        write_csv(folder / "frontier_2026_final_benchmark.csv", run["summary"], SUMMARY_COLUMNS)
        write_csv(folder / "model_rankings.csv", run["rankings"], RANKING_COLUMNS)
        report = "REFERENCE REPLAY: archived notebook output, not a new measurement.\n\n" + run["recorded_report"]
        (folder / "report.txt").write_text(report, encoding="utf-8")
        reports.append(f"Requested N={size:,}\n" + report)
        for row in run["summary"]:
            summaries.append(dict(Sample_Size=size, Repository=config["models"][row["Model"]][0],
                                  Pairs=run["capacities"][row["Domain"]], **row))
        for row in run["rankings"]:
            rankings.append(dict(Sample_Size=size, Repository=config["models"][row["Model"]][0],
                                 Pairs=sum(run["capacities"].values()), **row))
    write_csv(output / "benchmark_summary.csv", summaries, ["Sample_Size", *SUMMARY_COLUMNS, "Repository", "Pairs"])
    write_csv(output / "model_rankings.csv", rankings, ["Sample_Size", *RANKING_COLUMNS, "Repository", "Pairs"])
    write_csv(output / "frontier_2026_final_benchmark.csv", known[max(sizes)]["summary"], SUMMARY_COLUMNS)
    (output / "report.txt").write_text("\n\n".join(reports), encoding="utf-8")
    for image in (Path(__file__).resolve().parents[1] / "reference").glob("fig*_notebook_display.png"):
        shutil.copyfile(image, output / image.name)
    if not no_plots:
        import pandas as pd
        from .plots import save_figures
        save_figures(pd.DataFrame(summaries), pd.DataFrame(rankings), output)
    manifest = dict(status="reference_replay", mode="reference", config=config, sample_sizes=sizes,
                    notebook_sha256=reference["notebook_sha256"], provenance=reference["provenance"],
                    comparison=compare_reference(summaries, rankings, output), recomputed=False)
    manifest["artifacts"] = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in output.rglob("*") if path.is_file() and path.name not in ("run_manifest.json", "run.log")}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\n\n".join(reports))
    print(f"[REFERENCE] Replayed the saved notebook outputs into {output}.")
