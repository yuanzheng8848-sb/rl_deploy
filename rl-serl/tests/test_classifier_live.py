#!/usr/bin/env python3
"""Live OpenArm classifier reward smoke test.

This uses the same task config and wrapper stack as the actor:
    config.get_environment(fake_env=False, classifier=True)

The displayed reward is exactly the reward returned by env.step(...), i.e. the
same value the actor stores during training. It is not recomputed in this script.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


RL_SERL_ROOT = Path(__file__).resolve().parents[1]
for path in (
    RL_SERL_ROOT / "examples",
    RL_SERL_ROOT / "rl_robot_infra",
    RL_SERL_ROOT / "rl_launcher",
):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

import compat  # noqa: E402,F401
from experiments.mappings import CONFIG_MAPPING  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Live classifier reward + camera display.")
    parser.add_argument("--exp_name", default="openarm_pickplace")
    parser.add_argument("--window", default="rl-serl classifier live")
    parser.add_argument("--panel-width", type=int, default=384)
    parser.add_argument("--panel-height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--reset-on-done", action="store_true")
    return parser.parse_args()


def image_from_obs(obs, key):
    img = obs.get(key)
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 3 or img.shape[-1] != 3:
        return None
    return img.astype(np.uint8)


def to_display_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def panel_for_image(name, img, width, height, reward, done):
    if img is None:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        panel = cv2.resize(to_display_bgr(img), (width, height))
    color = (0, 220, 0) if float(reward) > 0 else (0, 0, 255)
    cv2.putText(
        panel,
        name,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"reward={float(reward):.3f}",
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"done={bool(done)}",
        (12, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return panel


def main():
    args = parse_args()
    config = CONFIG_MAPPING[args.exp_name]()
    env = config.get_environment(fake_env=False, classifier=True)
    image_keys = list(getattr(config, "image_keys", ["image_primary", "image_left", "image_right"]))
    classifier_keys = list(getattr(config, "classifier_keys", ["image_primary"]))

    print(f"Exp: {args.exp_name}")
    print(f"Policy image keys: {image_keys}")
    print(f"Classifier keys: {classifier_keys}")
    print(f"Classifier threshold: {getattr(config, 'classifier_threshold', None)}")
    print(f"Classifier ckpt: {getattr(config, 'classifier_ckpt_path', None)}")
    print("Reward shown here is env.step(...) reward from the actor wrapper path.")
    print("Press q or ESC to quit.")

    obs, _ = env.reset()
    reward = 0.0
    done = False
    delay = max(1, int(1000 / max(args.fps, 1.0)))

    try:
        while True:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, done, truncated, info = env.step(action)
            done = bool(done or truncated)

            panels = []
            for key in image_keys:
                panels.append(
                    panel_for_image(
                        key,
                        image_from_obs(obs, key),
                        args.panel_width,
                        args.panel_height,
                        reward,
                        done,
                    )
                )
            cv2.imshow(args.window, np.concatenate(panels, axis=1))
            print(
                f"reward={float(np.asarray(reward)):.3f} "
                f"done={done} succeed={info.get('succeed') if isinstance(info, dict) else None}"
            )

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord("q")):
                break
            if done:
                if args.reset_on_done:
                    obs, _ = env.reset()
                    reward = 0.0
                    done = False
                else:
                    time.sleep(0.2)
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
