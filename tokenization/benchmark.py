import csv
import hashlib
import json
import logging

from .data import extract_pair


FIELDS = ["Domain", "Model", "Repository", "Pair", "EN_Words", "BN_Words",
          "EN_Tokens", "BN_Tokens", "EN_Tokens/Word", "BN_Tokens/Word", "Compute_Disparity"]


def pair_metrics(en_tokens, bn_tokens, en_words, bn_words):
    """The notebook rounds each observation BEFORE averaging observations."""
    return (round(en_tokens / en_words, 2), round(bn_tokens / bn_words, 2),
            round(bn_tokens / en_tokens, 2) if en_tokens > 0 else 0)


def archived_pairs(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def evaluate(config, tokenizers, repositories, sizes, output, token, revisions, pairs_dir=None):
    summaries, rankings, counts = [], [], {}
    totals = {size: {} for size in sizes}
    archive = output / "pairs"
    archive.mkdir(exist_ok=True)
    with (output / "raw_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for domain_index, (domain, spec) in enumerate(config["domains"].items(), 1):
            filename = f"domain_{domain_index}.jsonl"
            destination = archive / filename
            source = pairs_dir / filename if pairs_dir is not None else None
            if source is not None and source.resolve() == destination.resolve():
                raise ValueError("Use a different --output when replaying archived pairs.")
            count = 0
            # Integer hundredths prevent accumulation drift across platforms.
            sums = {name: [0, 0, 0] for name in tokenizers}
            digest = hashlib.sha256()
            sample_hashes = {}
            captured = set()

            def snapshot(size):
                if not count:
                    return
                sample_hashes[str(size)] = digest.hexdigest()
                for name, values in sums.items():
                    summaries.append({"Sample_Size": size, "Domain": domain, "Model": name,
                                      "Repository": repositories[name], "Pairs": count,
                                      "Avg_EN_Density": values[0] / (100 * count),
                                      "Avg_BN_Density": values[1] / (100 * count),
                                      "Disparity_Multiplier": values[2] / (100 * count)})
                    total = totals[size].setdefault(name, [0, 0])
                    total[0] += values[2]
                    total[1] += count

            print(f"\n[STREAM] Accessing {'archived pairs' if source else 'network pipe'} for {domain}...")
            try:
                if source is not None:
                    stream = archived_pairs(source)
                else:
                    from datasets import load_dataset
                    stream = load_dataset(spec["path"], spec.get("config"), split=spec["split"],
                                          streaming=True, token=token, revision=revisions[spec["path"]])
                with destination.open("w", encoding="utf-8", newline="\n") as pairs_handle:
                    for item in stream:
                        en, bn = extract_pair(item)
                        en_words, bn_words = len(en.split()), len(bn.split())
                        if not en_words or not bn_words:
                            continue
                        metrics = {}
                        for name, tokenizer in tokenizers.items():
                            en_tokens = len(tokenizer.encode(en, add_special_tokens=False))
                            bn_tokens = len(tokenizer.encode(bn, add_special_tokens=False))
                            metrics[name] = (en_tokens, bn_tokens)
                        count += 1
                        pair = json.dumps(dict(pair=count, en=en, bn=bn), ensure_ascii=False, separators=(",", ":")) + "\n"
                        pairs_handle.write(pair)
                        digest.update(pair.encode("utf-8"))
                        for name, (en_tokens, bn_tokens) in metrics.items():
                            values = pair_metrics(en_tokens, bn_tokens, en_words, bn_words)
                            sums[name] = [a + int(round(100 * b)) for a, b in zip(sums[name], values)]
                            writer.writerow(dict(zip(FIELDS, [domain, name, repositories[name], count, en_words, bn_words,
                                                               en_tokens, bn_tokens, *values])))
                        if count in sizes:
                            snapshot(count)
                            captured.add(count)
                        if count % 1000 == 0:
                            logging.info("%s: %s pairs", domain, count)
                        if count >= max(sizes):
                            break
                counts[domain] = {"pairs": count}
                print(f" -> [SUCCESS] Finished pipeline execution for {domain}")
            except Exception as exc:
                counts[domain] = {"pairs": count, "error": str(exc)}
                logging.warning("Dataset %s stopped at %s pairs: %s", domain, count, exc)
            for size in sizes:
                if size not in captured:
                    snapshot(size)
            counts[domain].update(archive=f"pairs/{filename}", sha256=digest.hexdigest(), sample_hashes=sample_hashes,
                                  repository=spec["path"], revision=revisions[spec["path"]],
                                  source="archived_pairs" if source else "hub")
    for size, models in totals.items():
        for name, (total, count) in models.items():
            rankings.append({"Sample_Size": size, "Model": name, "Repository": repositories[name],
                             "Pairs": count, "Grand_Multiplier": total / (100 * count)})
    return summaries, rankings, counts
