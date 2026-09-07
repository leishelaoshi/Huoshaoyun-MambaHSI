#!/usr/bin/env python3
from __future__ import annotations

import argparse

from huoshaoyun_mambahsi.config import load_config
from huoshaoyun_mambahsi.workflow import stage_build_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_protocol.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    stage_build_samples(cfg)


if __name__ == "__main__":
    main()
