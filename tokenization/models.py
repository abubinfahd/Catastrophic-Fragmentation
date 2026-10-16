import logging

from transformers import AutoTokenizer, PreTrainedConfig


def load_tokenizers(models, token, trust_remote_code=False, revisions=None):
    tokenizers, repositories, failures = {}, {}, {}
    print("[INIT] Connecting to 2026 Grand Frontier Cluster Registry...")
    for name, paths in models.items():
        for path in paths:
            try:
                kwargs = {"revision": revisions[path]} if revisions is not None else {}
                if path == "unsloth/Llama-4-Scout-17B-16E-Instruct":
                    # Only tokenize text: avoid Llama4Config's unrelated RoPE
                    # validation while retaining the repository's tokenizer.
                    config_dict, _ = PreTrainedConfig.get_config_dict(path, token=token, **kwargs)
                    kwargs["config"] = PreTrainedConfig.from_dict(config_dict)
                tokenizer = AutoTokenizer.from_pretrained(
                    path, token=token, use_fast=True, trust_remote_code=trust_remote_code,
                    **kwargs,
                )
                tokenizers[name] = tokenizer
                repositories[name] = path
                logging.info("Loaded %s from %s", name, path)
                print(f" -> [LOADED] {name} synchronized via {path}")
                break
            except Exception as exc:
                failures.setdefault(name, []).append({"repository": path, "error": str(exc)})
                logging.warning("Could not load %s from %s: %s", name, path, exc)
    return tokenizers, repositories, failures
