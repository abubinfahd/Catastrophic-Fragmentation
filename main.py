"""Run the notebook benchmark unchanged, then add statistical tests."""
import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from benchmark import evaluate, load_tokenizers
from plots import save_figures
from statistical_analysis import analyze

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-sizes', type=int, nargs='+', help='Default: notebook scales 10000 50000 100000')
    parser.add_argument('--output', type=Path, help='Default: outputs/run_<timestamp>')
    parser.add_argument('--pairs-dir', type=Path, help='Optional saved original-notebook pairs instead of network streaming')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text())
    sizes = sorted(set(args.sample_sizes or config['sample_sizes']))
    if not sizes or min(sizes) < 2:
        parser.error('Statistical tests require sample sizes of at least 2')
    if args.pairs_dir and (args.pairs_dir / 'input_manifest.json').exists():
        source = json.loads((args.pairs_dir / 'input_manifest.json').read_text())
        if source.get('protocol') != 'notebook':
            parser.error('Use original-notebook pairs; the strict manuscript sample is a different experiment')
    output = args.output or ROOT / 'outputs' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new output folder to preserve existing results and logs')
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'run.log', mode='a', encoding='utf-8')], force=True)
    for noisy in ['httpx', 'httpcore', 'fontTools']:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv(ROOT / '.env')
    lock = json.loads((ROOT / 'repositories.lock.json').read_text())
    shutil.copy2(ROOT / 'config.json', output / 'config.json')
    shutil.copy2(ROOT / 'repositories.lock.json', output / 'repositories.lock.json')
    manifest = {'status': 'running', 'sample_sizes': sizes, 'seed': config['seed'],
                'source_notebook': '../Reviewer_Audit_2026-10-21/Paper_Token.ipynb',
                'method': 'Original notebook extraction and acceptance; Python round(2) per pair before arithmetic means. No added NFC, script thresholds, three-word minimum or geometric aggregation.',
                'statistical_assumptions': 'Tests use the same rounded per-pair disparities. Model tests are paired; corpus tests are independent Welch tests. CIs assume independent rows; corpus/document dependencies and fixed-prefix sampling limit inference.',
                'pairs_source': str(args.pairs_dir) if args.pairs_dir else 'pinned Hugging Face streams'}

    def save_manifest():
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    save_manifest()
    try:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        os.environ['PYTHONHASHSEED'] = str(config['seed'])
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        tokenizers = load_tokenizers(config, lock)
        for size in sizes:
            random.seed(config['seed'])
            np.random.seed(config['seed'])
            folder = output / f'n_{size}'
            raw, summary, ranking = evaluate(config, tokenizers, size, folder, lock, args.pairs_dir)
            analyze(raw, folder)
            save_figures(summary, ranking, folder / 'figures', size, config)
            logging.info('N=%s summary:\n%s\nOverall arithmetic ranking:\n%s', size,
                         summary.to_string(index=False), ranking.to_string(index=False))
        manifest['status'] = 'complete'
        logging.info('Complete. Tables, per-pair data, statistics and figures: %s', output.resolve())
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        logging.exception('Run failed; existing partial results and logs retained')
        raise
    finally:
        save_manifest()


if __name__ == '__main__':
    main()
