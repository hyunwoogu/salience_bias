"""Load (block, imgn-experiment, target) -> on-disk image filename from per-dataset YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_imgn_exp_to_data(dataset_name: str) -> dict[tuple[int, int, str], str]:
    config_path = _CONFIG_DIR / f"{dataset_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No image-map config for dataset {dataset_name!r}: {config_path}"
        )

    cfg = yaml.safe_load(config_path.read_text())
    n_trials = int(cfg.get("n_trials", 100))
    out: dict[tuple[int, int, str], str] = {}

    for rule in cfg["rules"]:
        if "blocks" in rule:
            blocks = [int(b) for b in rule["blocks"]]
        else:
            blocks = [int(rule["block"])]
        target = rule["target"]
        offset = int(rule.get("expn_offset", 0))
        template = rule["file"]
        for block in blocks:
            for expn in range(n_trials):
                idx = expn + offset
                out[(block, expn, target)] = template.format(expn=expn, idx=idx)

    return out
