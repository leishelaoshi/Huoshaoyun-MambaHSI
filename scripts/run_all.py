#!/usr/bin/env python3
from __future__ import annotations

import argparse

from huoshaoyun_mambahsi.config import load_config
from huoshaoyun_mambahsi.workflow import run_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_protocol.yaml")
    args = p.parse_args()
    run_all(load_config(args.config))


if __name__ == "__main__":
    main()
