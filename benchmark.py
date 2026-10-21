"""Paper_Token.ipynb evaluation logic, with pair IDs retained for statistical tests."""
import csv
import gc
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd


def extract_pair(item):
    """Same dictionary traversal, stripping and Unicode selection as _auto_extract."""
    values = []

    def extract_vals(value):
        if isinstance(value, dict):
            for nested in value.values():
                extract_vals(nested)
        elif isinstance(value, str):
            values.append(value)

    extract_vals(item)
    en_text, bn_text = '', ''
    for val in values:
        val = str(val).strip()
        if not val:
            continue
        if not bn_text and re.search(r'[\u0980-\u09FF]', val):
            bn_text = val
        elif not en_text and re.search(r'[a-zA-Z]', val) and not re.search(r'[\u0980-\u09FF]', val):
            en_text = val
    return en_text, bn_text


def pair_metrics(en_tokens, bn_tokens, en_words, bn_words):
    """Keep the notebook's Python round-before-mean calculations unchanged."""
    return (round(en_tokens / en_words, 2), round(bn_tokens / bn_words, 2),
            round(bn_tokens / en_tokens, 2) if en_tokens > 0 else 0)


def load_tokenizers(config, lock):
    from transformers import AutoTokenizer, PreTrainedConfig
    tokenizers = {}
    token = os.environ.get('HF_TOKEN') or None
    for alias, paths in config['models'].items():
        repo = paths[0]
        revision = lock.get('models', {}).get(repo, {}).get('revision')
        cached = Path.home() / '.cache/huggingface/hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / str(revision)
        source = str(cached) if (cached / 'tokenizer.json').exists() else repo
        options = {'use_fast': True, 'trust_remote_code': False, 'token': token}
        if source == repo and revision:
            options['revision'] = revision
        if repo == 'unsloth/Llama-4-Scout-17B-16E-Instruct':
            # Avoid neural RoPE validation while using the exact same fast tokenizer.
            details, _ = PreTrainedConfig.get_config_dict(source, token=token,
                                                        **({'revision': revision} if source == repo and revision else {}))
            options['config'] = PreTrainedConfig.from_dict(details)
        tokenizers[alias] = AutoTokenizer.from_pretrained(source, **options)
        logging.info('Loaded %s from %s', alias, repo)
    return tokenizers


def evaluate(config, tokenizers, sample_limit, folder, lock, pairs_dir=None):
    """Same stream order, acceptance rule and per-model encode calls as the notebook."""
    folder.mkdir(parents=True, exist_ok=True)
    fields = ['Domain', 'Model', 'Pair', 'EN_Tokens/Word', 'BN_Tokens/Word', 'Compute_Disparity']
    raw_path = folder / 'raw_metrics.csv'
    loads = {}
    with raw_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (domain, spec) in enumerate(config['domains'].items(), 1):
            logging.info('Evaluating %s: target %s pairs', domain, sample_limit)
            source_file = None
            if pairs_dir:
                source_file = (pairs_dir / f'domain_{index}/pairs.jsonl').open(encoding='utf-8')
                dataset = (json.loads(line) for line in source_file)
            else:
                from datasets import load_dataset
                revision = lock.get('datasets', {}).get(spec['path'], {}).get('revision')
                dataset = load_dataset(spec['path'], spec['config'], split=spec['split'], streaming=True,
                                       token=os.environ.get('HF_TOKEN') or None,
                                       **({'revision': revision} if revision else {}))
            count = 0
            records = []
            try:
                for item in dataset:
                    if count >= sample_limit:
                        break
                    en_text, bn_text = extract_pair(item)
                    if not en_text or not bn_text:
                        continue
                    en_words, bn_words = len(en_text.split()), len(bn_text.split())
                    if en_words == 0 or bn_words == 0:
                        continue
                    for alias, tokenizer in tokenizers.items():
                        en_tokens = len(tokenizer.encode(en_text, add_special_tokens=False))
                        bn_tokens = len(tokenizer.encode(bn_text, add_special_tokens=False))
                        en_density, bn_density, disparity = pair_metrics(en_tokens, bn_tokens, en_words, bn_words)
                        records.append(dict(zip(fields, [domain, alias, count + 1, en_density, bn_density, disparity])))
                    count += 1
                    if count % 500 == 0:
                        writer.writerows(records)
                        handle.flush()
                        records.clear()
                        gc.collect()
                    if count % 10000 == 0:
                        logging.info('%s: %s pairs', domain, count)
                writer.writerows(records)
                handle.flush()
            finally:
                if source_file:
                    source_file.close()
            loads[domain] = count
            if count != sample_limit:
                raise RuntimeError(f'{domain}: only {count} of {sample_limit} pairs; partial raw data retained')
    # Identical groupby/mean/round operations to compile_final_report.
    raw = pd.read_csv(raw_path, memory_map=True)
    summary, ranking = summarize(raw, folder)
    (folder / 'actual_loads.json').write_text(json.dumps(loads, indent=2), encoding='utf-8')
    return raw, summary, ranking


def summarize(raw, folder):
    """The notebook's compile_final_report arithmetic, without deleting its input."""
    summary = raw.groupby(['Domain', 'Model']).agg(
        Avg_EN_Density=('EN_Tokens/Word', 'mean'), Avg_BN_Density=('BN_Tokens/Word', 'mean'),
        Disparity_Multiplier=('Compute_Disparity', 'mean')).reset_index().round(2)
    summary = summary.sort_values(['Domain', 'Disparity_Multiplier'], ascending=[True, False])
    ranking = raw.groupby('Model').agg(Grand_Multiplier=('Compute_Disparity', 'mean')).reset_index().round(2)
    ranking = ranking.sort_values('Grand_Multiplier', ascending=False)
    summary.to_csv(folder / 'frontier_2026_final_benchmark.csv', index=False)
    ranking.to_csv(folder / 'model_rankings.csv', index=False)
    return summary, ranking
