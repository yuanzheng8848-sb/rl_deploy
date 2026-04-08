#!/usr/bin/env python3
"""
使用非训练数据测试分类器：对 session_20260301_100225 的 cam_2_rgb 每一帧计算成功概率，
并生成曲线图与示例图像到当前目录（及子文件夹）。
"""
import os
import sys
import glob
import ctypes
import site
from pathlib import Path

# 保证可导入 serl_launcher（与 eval_classifier 一致从项目根跑时无需改 path）
_SCRIPT_DIR = Path(__file__).resolve().parent
_RL_DEPLOY = _SCRIPT_DIR.parents[1]  # rl_deploy
for _p in [_RL_DEPLOY, _SCRIPT_DIR.parents[3], _SCRIPT_DIR.parents[4]]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# serl_launcher 可能在 zy/serl/serl_launcher 下
_zy = _SCRIPT_DIR.parents[4] if _SCRIPT_DIR.parents[4].exists() else _SCRIPT_DIR.parents[3]
_serl_launcher = _zy / "serl" / "serl_launcher"
if _serl_launcher.exists() and str(_serl_launcher) not in sys.path:
    sys.path.insert(0, str(_serl_launcher))

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
from jax import numpy as jnp
import flax.linen as nn
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from serl_launcher.networks.reward_classifier import load_classifier_func

# --- JAX/CUDA Fix (与 eval_classifier 一致) ---
try:
    nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
    libs = [
        "cublas/lib", "cudnn/lib", "cufft/lib", "cusolver/lib",
        "cusparse/lib", "nccl/lib", "nvjitlink/lib"
    ]
    for lib in libs:
        path = os.path.join(nvidia_base, lib)
        if os.path.exists(path):
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld}"
    os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={nvidia_base}"
    nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
    if os.path.exists(nvjitlink_path):
        ctypes.CDLL(nvjitlink_path)
    cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
    if os.path.exists(cusparse_path):
        ctypes.CDLL(cusparse_path)
except Exception as e:
    print(f"[Warning] JAX/CUDA fix: {e}")

# 路径：使用当前脚本所在目录及子目录
TEST_IMAGE_DIR = _SCRIPT_DIR / "session_20260301_100225" / "images" / "cam_2_rgb"
CHECKPOINT_PATH = _RL_DEPLOY / "classifier" / "classifier_ckpt"
OUTPUT_DIR = _SCRIPT_DIR / "test_logs"
TEST_IMAGES_SUBDIR = _SCRIPT_DIR / "test_logs" / "example_images"


def fix_image_shape(x):
    if isinstance(x, np.ndarray):
        x = jnp.array(x)
    shape = x.shape
    if len(shape) == 3 and shape[0] == 128 and shape[1] == 128 and shape[2] == 3:
        x = jnp.expand_dims(x, axis=0)
    elif len(shape) == 4 and shape[0] == 1 and shape[1] == 128 and shape[2] == 128 and shape[3] == 3:
        pass
    else:
        x = x.reshape((1, 128, 128, 3))
    return x


def load_all_frames(image_dir):
    """加载目录下所有 jpg，按文件名排序，返回 (paths, images)。"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return [], []
    path_list = sorted(glob.glob(str(image_dir / "*.jpg")))
    paths, images = [], []
    for path in path_list:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        paths.append(path)
        images.append(img)
    return paths, images


def main():
    print("Test classifier on non-training data (every frame prob).")
    print(f"Image dir: {TEST_IMAGE_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Test images subdir: {TEST_IMAGES_SUBDIR}")

    paths, images = load_all_frames(TEST_IMAGE_DIR)
    if not images:
        print("No images found.")
        return

    print(f"Loaded {len(images)} frames.")

    # 加载分类器
    rng = jax.random.PRNGKey(0)
    init_img = fix_image_shape(images[0])
    sample = {"image_0": init_img, "state": jnp.zeros((1, 14))}
    try:
        classifier_func = load_classifier_func(
            key=rng,
            sample=sample,
            image_keys=["image_0"],
            checkpoint_path=os.path.abspath(CHECKPOINT_PATH),
        )
    except Exception as e:
        print(f"Failed to load classifier: {e}")
        return

    # 逐帧计算 prob
    probs = []
    for img in tqdm(images, desc="Inference"):
        img_jax = fix_image_shape(img)
        logits = classifier_func({"image_0": img_jax})
        prob = float(nn.sigmoid(logits).item())
        probs.append(prob)
    probs = np.array(probs)
    frame_indices = np.arange(len(probs))

    # 输出目录与子目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMAGES_SUBDIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 每帧概率曲线 ----------
    plt.figure(figsize=(12, 5))
    plt.plot(frame_indices, probs, color="tab:blue", linewidth=1.2, label="P(success)")
    plt.scatter(frame_indices, probs, color="red", s=15, alpha=0.8, zorder=3, label="Frames")
    plt.axhline(0.5, color="gray", linestyle="--", label="Threshold 0.5")
    plt.xlabel("Frame index")
    plt.ylabel("Probability (success)")
    plt.title("Classifier Probability per Frame (Non-Training Session)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_prob_per_frame.png", dpi=120)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'test_prob_per_frame.png'}")

    # ---------- 2. 概率分布直方图 ----------
    plt.figure(figsize=(8, 5))
    plt.hist(probs, bins=min(50, len(probs)), color="tab:blue", edgecolor="white", alpha=0.8)
    plt.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
    plt.xlabel("Probability (success)")
    plt.ylabel("Frame count")
    plt.title("Distribution of Per-Frame Success Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_prob_histogram.png", dpi=120)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'test_prob_histogram.png'}")

    # ---------- 3. 统计摘要（可选：简单文本或表格图） ----------
    n_success = int(np.sum(probs > 0.5))
    n_failure = len(probs) - n_success
    plt.figure(figsize=(6, 4))
    plt.bar(["Pred Success\n(prob>0.5)", "Pred Failure\n(prob≤0.5)"], [n_success, n_failure], color=["#2ecc71", "#e74c3c"])
    plt.ylabel("Frame count")
    plt.title("Frame-level Prediction Summary (Threshold=0.5)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_prediction_summary.png", dpi=120)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'test_prediction_summary.png'}")

    # ---------- 4. 保存若干测试图像到子文件夹 ----------
    # 选帧：首、尾、中间，以及概率最低/最高的几帧
    indices_to_save = set()
    indices_to_save.add(0)
    indices_to_save.add(len(images) - 1)
    indices_to_save.add(len(images) // 2)
    k = min(5, len(images) // 2)
    for i in np.argsort(probs)[:k]:
        indices_to_save.add(int(i))
    for i in np.argsort(probs)[-k:]:
        indices_to_save.add(int(i))
    for i in sorted(indices_to_save):
        if i >= len(images):
            continue
        img = images[i]
        prob = probs[i]
        pred = "success" if prob > 0.5 else "failure"
        fname = f"frame_{i:04d}_{pred}_{prob:.4f}.jpg"
        out_path = TEST_IMAGES_SUBDIR / fname
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 可选：在图上写 prob
        cv2.putText(img_bgr, f"P={prob:.3f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(out_path), img_bgr)
    print(f"Saved {len(indices_to_save)} sample images to: {TEST_IMAGES_SUBDIR}")

    print("\nDone. Charts and sample images are in test_logs/ and test_logs/example_images/.")


if __name__ == "__main__":
    main()
