#!/usr/bin/env python3
"""Inspect trajectory pickle files and report frame counts."""

from __future__ import annotations

import pickle
from pathlib import Path
BASE_DIR = Path("/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/collected")


def count_frames(path: Path) -> int:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("transitions", "trajectory", "frames"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
    raise ValueError(f"Unsupported pickle structure in {path}")


def get_counts(label: str) -> list[int]:
    return [
        count_frames(path)
        for path in sorted((BASE_DIR / label).glob("*.pkl"))
    ]


def main() -> int:
    success_counts = get_counts("success")

    if not success_counts:
        print("No valid trajectory pickles found.")
        return 1

    for i in range(0, len(success_counts), 5):
        chunk = success_counts[i : i + 5]
        print(" ".join(f"{value:>4d}" for value in chunk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
