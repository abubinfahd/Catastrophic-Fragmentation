# Reproducing Paper_Token.ipynb

This project follows the master notebook's five tokenizers, three datasets, seed 42, and first 10,000 / 50,000 / 100,000 valid sentence pairs per domain. It exports the notebook's tables and two journal figures, plus input archives, provenance, and an explicit numerical comparison.

Model labels match the five author-confirmed checkpoint IDs in `config.json`. Runs append diagnostics and printed result tables to `outputs/<mode>/run.log`; existing log content is preserved. Start/finish markers identify successive runs. Use a new output directory when you also need to preserve each run's separate CSV files and manifest.

## Choose the reproduction you need

| Goal | Command | What is reproduced |
| --- | --- | --- |
| Recover the saved paper tables and figures | `python main.py --mode reference` | Archived notebook values; no tokenization is recomputed |
| Independently recompute and compare with the paper | `python main.py --mode benchmark --verify-reference` | Fresh measurements from pinned tokenizers and datasets |
| Recover tables and original plot images without installing dependencies | `python main.py --mode reference --no-plots` | Archived tables and embedded notebook images |

**Exact archived-output replay is supported. Exact historical recomputation is not guaranteed:** the original notebook did not preserve its raw sentence pairs, package versions, or repository commits. Fresh runs report numerical differences and archive their inputs for future replication.

## Step 1: Obtain the project and open its directory

Extract `tokenization_project_reproducible.zip` or obtain the project folder. The package contains:

```text
Paper_Token.ipynb
tokenization_project/
    main.py
    config.json
    requirements.lock
    repositories.lock.json
    reference/
    tokenization/
    tests/
```

Open a terminal in the directory containing `tokenization_project`, then run:

```sh
cd tokenization_project
python --version
```

Use **Python 3.12**. All following commands run from inside `tokenization_project`; all file paths are relative to that directory. The project can be placed anywhere. If your system names its Python 3.12 executable `python3` or `py -3.12`, use that command until your environment is activated.

## Step 2: Prepare a Python environment

If you already have an activated Python 3.12 environment, keep using it and skip environment creation. Otherwise, create an environment:

```sh
python -m venv .venv
```

Activate it on **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on **macOS/Linux**:

```sh
source .venv/bin/activate
```

Install the locked dependencies:

```sh
python -m pip install --require-hashes -r requirements.lock
```

If you prefer uv, you can use `uv venv --python 3.12 .venv` for environment creation and `uv pip install --require-hashes -r requirements.lock` after activation. Use `requirements.lock` for reproduction; `requirements.txt` lists the unpinned direct dependencies for maintenance.

If activation is unavailable, invoke your environment's Python executable directly in subsequent commands: `.\.venv\Scripts\python.exe` on Windows or `./.venv/bin/python` on macOS/Linux. This also works with an existing environment stored elsewhere; no fixed machine path is required.

For archived tables and original plot images only, dependency installation is unnecessary. Run `python main.py --mode reference --no-plots` after Step 1.

## Step 3: Reproduce the saved paper output

```sh
python main.py --mode reference
```

This replays the notebook's saved 10,000-, 50,000-, and 100,000-pair tables and generates PNG/PDF figures from those values. It makes no Hub requests and downloads no model, tokenizer, or dataset files. Results are written to `outputs/reference/`.

Check `outputs/reference/run_manifest.json`. A successful replay contains:

```json
{
  "status": "reference_replay",
  "recomputed": false,
  "comparison": {
    "status": "match",
    "compared_values": 150,
    "different_values": 0
  }
}
```

This is an excerpt of the manifest. Inspect `outputs/reference/n_10000/`, `n_50000/`, and `n_100000/` for the three result tables and rankings. The notebook's embedded plot images, with checkpoint labels corrected on 11 October 2026, are copied into the output directory. This mode recovers recorded results; it does not independently validate the experiment.

## Step 4: Prepare an independent benchmark

Keep `config.json`, `requirements.lock`, and `repositories.lock.json` unchanged when comparing against this baseline. The dependency lock pins package versions and file hashes; the repository lock pins all five models and three datasets to immutable commits. These are the project's replication baseline, not recovered historical notebook versions.

If authentication is required, create `.env` from the example on **Windows PowerShell**:

```powershell
Copy-Item .env.example .env
```

Or on **macOS/Linux**:

```sh
cp .env.example .env
```

Skip the copy if `.env` already exists. Edit `.env` locally and replace the placeholder with your own Hugging Face token:

```dotenv
HF_TOKEN=hf_your_token_here
```

The file is loaded automatically; an existing environment variable takes precedence. Repository access conditions must be accepted on the account associated with the token when required. Keep `.env` private; it is excluded from the researcher package.

Check repository URLs and run the offline validation suite:

```sh
python check_model_links.py
python -X utf8 -m unittest discover -s tests -v
```

The URL check retrieves metadata only. The tests use archived outputs and local fixtures; neither command downloads model weights, tokenizer files, or dataset content. Offline tests should finish with `OK`; the suite contains seven tests.

## Step 5: Run the independent benchmark

Set the seed before starting Python on **Windows PowerShell**:

```powershell
$env:PYTHONHASHSEED = '42'
python main.py --mode benchmark --verify-reference
```

Or on **macOS/Linux**:

```sh
PYTHONHASHSEED=42 python main.py --mode benchmark --verify-reference
```

This processes the first valid sentence pairs in dataset order and measures all five tokenizers at every requested sample size. It streams dataset content and downloads tokenizer assets when they are not cached. **No model weights or GPU are required.** Hugging Face uses its normal cache outside the project; set `HF_HOME` before starting Python if you want a different cache location.

Results are written to `outputs/benchmark/`. Before a full run, you can check connectivity and processing with a smaller diagnostic:

```sh
python main.py --mode benchmark --sample-sizes 100 --output outputs/diagnostic
```

There is no saved 100-pair notebook reference, so this diagnostic's comparison is marked `unavailable`. It is not a substitute for the three full sample sizes. No live full benchmark was run while preparing this project; validation used saved outputs and local fixtures.

## Step 6: Verify and share the results

For an independent run, inspect `outputs/benchmark/run_manifest.json` and `outputs/benchmark/reference_comparison.csv`:

| Check | Successful result | If it fails |
| --- | --- | --- |
| Manifest `status` | `complete` | Read `error`, dataset capacities, and `run.log` |
| Actual dataset capacities | 100,000 valid pairs in each domain for the default run | The default configuration rejects incomplete samples |
| `reference_comparison.status` | `match` | Review expected and observed values in the comparison CSV |
| Default comparison totals | 150 values compared, zero differences | Fresh values do not exactly match the saved notebook |

`--verify-reference` returns a nonzero exit code on a mismatch or an unavailable comparison, retaining fresh results and diagnostics. Without this flag, an otherwise complete benchmark keeps its results and records any differences. Do not describe a reference replay or a mismatched benchmark as a successful independent replication.

Preserve the complete output folder, source code, `config.json`, both lock files, and the master notebook when publishing a run. The output manifest identifies the versions, revisions, input hashes, and generated artifacts. Share any required repository access instructions separately; never include `.env` or a token.

## Step 7: Recompute from archived sentence pairs

A completed benchmark archives its ordered inputs under `outputs/benchmark/pairs/`. Reuse those exact pairs with a new output directory:

```sh
python main.py --mode benchmark --pairs-dir outputs/benchmark/pairs --output outputs/recomputed --verify-reference
```

Set `PYTHONHASHSEED=42` as in Step 5. The command avoids dataset streaming, but tokenizer files must be cached or accessible on the Hub. For another researcher's run, place their published pairs in a local directory and replace the relative `--pairs-dir` value accordingly. Keep their configuration, repository lock, dependency lock, and manifest with the inputs.

Compare ordered pair hashes, token counts, and result values with the published run. `--verify-reference` compares with the original notebook's saved tables, not automatically with another run's manifest. Use a new output directory for every independent run, and preserve dataset attribution and applicable terms with shared inputs.

## Models and datasets

| Notebook display label | Exact tokenizer repository |
| --- | --- |
| Alibaba (Qwen3 1.7B) | `Qwen/Qwen3-1.7B` |
| Meta (Llama 4 Scout) | `unsloth/Llama-4-Scout-17B-16E-Instruct` |
| Google (Gemma 4 E4B-it) | `google/gemma-4-E4B-it` |
| Microsoft (Phi-4-mini-instruct) | `microsoft/Phi-4-mini-instruct` |
| Mistral (7B Instruct v0.3) | `mistralai/Mistral-7B-Instruct-v0.3` |

The display labels identify the actual checkpoints confirmed by the author. Model names were corrected on 11 October 2026; recorded metric values and repository IDs were preserved. No fallback model substitutions are allowed.

| Domain | Dataset | Configuration | Split |
| --- | --- | --- | --- |
| OPUS-100 (General) | `Helsinki-NLP/opus-100` | `bn-en` | `train` |
| Samanantar (Formal Literature) | `ai4bharat/samanantar` | `bn` | `train` |
| Web & Wiki (Domain 3) | `sentence-transformers/parallel-sentences-ccmatrix` | `en-bn` | `train` |

## Calculation protocol

Dictionary traversal and Unicode selection match the notebook's `_auto_extract`: the first string containing Bengali characters is Bengali; the first string containing Latin letters and no Bengali characters is English. Lists are ignored, matching the notebook. Text is stripped but otherwise not normalized, shuffled, filtered for translation quality, or deduplicated. This retains the notebook's corpus noise and extraction assumptions.

For each valid pair, word counts use `len(text.split())` and token counts use `tokenizer.encode(text, add_special_tokens=False)`. All five tokenizers process the same ordered pairs. A sample limit counts sentence pairs, not model rows.

- `EN_Tokens/Word = round(EN_Tokens / EN_Words, 2)`.
- `BN_Tokens/Word = round(BN_Tokens / BN_Words, 2)`.
- `Compute_Disparity = round(BN_Tokens / EN_Tokens, 2)`; zero English tokens produce zero disparity, as in the notebook.
- Domain metrics average those already rounded observations, then round to two decimals.
- The grand multiplier is the **arithmetic mean** of all per-pair disparities across domains, weighted by actual pair counts. It is not a geometric mean or a ratio of average word densities.

Integer hundredths accumulate the notebook-rounded observations without floating-point summation drift. Final rounding uses pandas, as in the notebook. Domain and grand rankings run from worst to best. The Llama tokenizer uses a generic configuration to bypass model-only RoPE validation while retaining its pinned tokenizer assets. Remote repository code is disabled.

All configured tokenizers and full requested dataset capacities are required by default. A stream error or short domain marks the run failed, retains diagnostics, and does not publish a complete benchmark table. `require_full_samples: false` can explicitly permit partial datasets; actual counts remain disclosed.

## Outputs and provenance

| Artifact | Contents |
| --- | --- |
| `n_10000/`, `n_50000/`, `n_100000/` | Notebook-schema `frontier_2026_final_benchmark.csv`, grand `model_rankings.csv`, and readable `report.txt` |
| `frontier_2026_final_benchmark.csv` | Largest requested sample's notebook-schema domain table |
| `benchmark_summary.csv`, `model_rankings.csv` | Combined results with sample sizes, actual counts, and repositories |
| `fig1_cross_domain_disparity.png` / `.pdf` | Grouped domain disparity figure, from the largest requested sample |
| `fig2_grand_inefficiency_ranking.png` / `.pdf` | Grand ranking figure, from the largest requested sample |
| `reference_comparison.csv` | Expected and observed values with explicit equality checks |
| `run_manifest.json`, `run.log`, `report.txt` | Run mode, completion status, configuration, and diagnostics |
| `raw_metrics.csv` (benchmark mode) | Pair/model token counts and the notebook-rounded observations |
| `pairs/domain_<index>.jsonl` (benchmark mode) | Exact processed text pairs in configuration domain order |
| `fig1_notebook_display.png`, `fig2_notebook_display.png` (reference mode) | Copies of the notebook's embedded plots with the corrected checkpoint labels |

The generated journal figures use the notebook's layout and palette. Their labels describe arithmetic averaging and token-count ratios; the embedded display plots retain the notebook's statistical terminology and now use the corrected checkpoint identities. Tables preserve the saved numerical results. Rendered PDF/PNG bytes can depend on fonts and platform. The pre-correction notebook and plots remain available in the manuscript-work backups.

Fresh manifests record Python/package versions, pinned revisions, source and dependency-lock hashes, actual capacities, ordered pair hashes at every sample limit, and output file hashes. Keep the entire output folder when publishing results. Researchers can reuse the archived pairs, avoiding dependence on Hub streaming order:

```sh
python main.py --mode benchmark --pairs-dir outputs/benchmark/pairs --output outputs/recomputed --verify-reference
```

Use a new output directory for each independent run. Compare pair hashes and token counts as well as rounded tables when assessing reproduction. Dataset source IDs and revisions remain in the manifest; preserve their attribution and applicable dataset terms with shared inputs.

