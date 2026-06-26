#!/usr/bin/env python3
"""Generate primary-camera center-crop comparisons from recorded demos."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


RL_DEPLOY_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = RL_DEPLOY_DIR / "demo" / "collected"
RAW_IMAGE_DIR = DEMO_DIR / "raw_images"
OUTPUT_DIR = RL_DEPLOY_DIR / "output"
MODEL_IMAGE_SIZE = (128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save image_primary crop previews under rl_deploy/output."
    )
    parser.add_argument(
        "--label",
        choices=("success", "failure", "all"),
        default="success",
        help="Recorded demo split to sample from.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of source frames to sample.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.3],
        help="Centered crop side ratios to compare.",
    )
    parser.add_argument(
        "--y-offsets",
        type=float,
        nargs="+",
        default=[0.0],
        help="Vertical crop center offsets; positive values move the crop down.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "primary_crop_tests",
        help="Directory for generated images.",
    )
    return parser.parse_args()


def crop_rgb_image(
    img: np.ndarray,
    crop_ratio: float,
    y_offset_ratio: float = 0.0,
) -> np.ndarray:
    img = np.asarray(img)
    ratio = float(np.clip(crop_ratio, 0.05, 1.0))
    y_offset_ratio = float(np.clip(y_offset_ratio, -0.5, 0.5))
    h, w = img.shape[:2]
    crop_h = max(1, int(round(h * ratio)))
    crop_w = max(1, int(round(w * ratio)))
    center_y = (h / 2.0) + (h * y_offset_ratio)
    y0 = int(round(center_y - crop_h / 2.0))
    y0 = int(np.clip(y0, 0, max(0, h - crop_h)))
    x0 = max(0, (w - crop_w) // 2)
    cropped = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, MODEL_IMAGE_SIZE)


def list_raw_primary_frames(label: str) -> list[Path]:
    labels = ("success", "failure") if label == "all" else (label,)
    frames: list[Path] = []
    for split in labels:
        frames.extend(sorted((RAW_IMAGE_DIR / split).glob("*/image_primary/frame_*.png")))
    return sorted(frames)


def read_raw_frame(path: Path) -> np.ndarray | None:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def extract_pkl_primary_images(label: str) -> list[tuple[str, np.ndarray]]:
    labels = ("success", "failure") if label == "all" else (label,)
    images: list[tuple[str, np.ndarray]] = []
    for split in labels:
        for pkl_path in sorted((DEMO_DIR / split).glob("*.pkl")):
            with pkl_path.open("rb") as handle:
                transitions = pickle.load(handle)
            if not isinstance(transitions, list):
                continue
            for idx, transition in enumerate(transitions):
                next_obs = transition.get("next_observations", {})
                img = next_obs.get("image_primary")
                if img is None:
                    continue
                img = np.asarray(img)
                if img.shape == (1, 128, 128, 3):
                    img = img[0]
                if img.shape == (128, 128, 3):
                    images.append((f"{pkl_path.stem}_frame_{idx:06d}", img.astype(np.uint8)))
    return images


def pick_evenly(items: list, count: int) -> list:
    if count <= 0 or len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(idx)] for idx in indices]


def draw_label(img_rgb: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(
        canvas,
        text,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def make_contact_sheet(
    source_name: str,
    img_rgb: np.ndarray,
    ratios: list[float],
    y_offsets: list[float],
) -> np.ndarray:
    panels = []
    original = cv2.resize(img_rgb, MODEL_IMAGE_SIZE)
    panels.append(draw_label(original, "source"))
    for ratio in ratios:
        for y_offset in y_offsets:
            cropped = crop_rgb_image(img_rgb, ratio, y_offset_ratio=y_offset)
            panels.append(draw_label(cropped, f"c{ratio:.2f} y{y_offset:+.2f}"))
    return np.hstack(panels)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = pick_evenly(list_raw_primary_frames(args.label), args.num_frames)
    samples: list[tuple[str, np.ndarray]] = []
    for frame_path in raw_frames:
        img = read_raw_frame(frame_path)
        if img is not None:
            samples.append((frame_path.parent.parent.name + "_" + frame_path.stem, img))

    if not samples:
        samples = pick_evenly(extract_pkl_primary_images(args.label), args.num_frames)

    if not samples:
        print("No image_primary frames found in recorded data.")
        return 1

    all_sheets = []
    for sample_idx, (name, img_rgb) in enumerate(samples):
        sheet = make_contact_sheet(name, img_rgb, args.ratios, args.y_offsets)
        all_sheets.append(sheet)
        save_rgb(args.output_dir / f"{sample_idx:02d}_{name}_sheet.png", sheet)

        per_frame_dir = args.output_dir / f"{sample_idx:02d}_{name}"
        save_rgb(per_frame_dir / "source_resized.png", cv2.resize(img_rgb, MODEL_IMAGE_SIZE))
        for ratio in args.ratios:
            for y_offset in args.y_offsets:
                save_rgb(
                    per_frame_dir / f"crop_{ratio:.2f}_y_{y_offset:+.2f}.png",
                    crop_rgb_image(img_rgb, ratio, y_offset_ratio=y_offset),
                )

    combined = np.vstack(all_sheets)
    save_rgb(args.output_dir / "all_crop_comparisons.png", combined)

    print(f"Saved {len(samples)} crop comparison sheets to: {args.output_dir}")
    print(f"Combined preview: {args.output_dir / 'all_crop_comparisons.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
