#!/usr/bin/env python3
"""Deep-merge a Forge leaf config (inherit chain via load_config) with override YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.utils.config import _deep_merge, load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", help="Leaf config path (inherits resolved by load_config)")
    p.add_argument("overrides", help="YAML file merged on top of the resolved config")
    p.add_argument("out", help="Merged config path to write")
    args = p.parse_args()

    base_path = Path(args.base).resolve()
    over_path = Path(args.overrides).resolve()
    out_path = Path(args.out).resolve()

    merged = load_config(str(base_path))
    extra = yaml.safe_load(over_path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(merged, extra)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
