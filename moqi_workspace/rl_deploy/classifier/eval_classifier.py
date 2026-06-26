import ctypes
import glob
import os
import pickle as pkl
import site

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import cv2
import flax.linen as nn
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import numpy as jnp
from tqdm import tqdm

from serl_launcher.networks.reward_classifier import load_classifier_func

plt.switch_backend("Agg")


SUCCESS_DIR = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/collected/success"
CHECKPOINT_PATH = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/classifier_ckpt"
IMAGE_KEY = "image_primary"
SUCCESS_TAIL_FRAMES_PER_TRAJ = 30
FAILURE_HEAD_FRAMES_PER_TRAJ = 30
HOLDOUT_TRAJ_COUNT = 3


try:
    nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
    for lib in (
        "cublas/lib",
        "cudnn/lib",
        "cufft/lib",
        "cusolver/lib",
        "cusparse/lib",
        "nccl/lib",
        "nvjitlink/lib",
    ):
        path = os.path.join(nvidia_base, lib)
        if os.path.exists(path):
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld}"
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={nvidia_base}"

    nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
    cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
    if os.path.exists(nvjitlink_path):
        ctypes.CDLL(nvjitlink_path)
    if os.path.exists(cusparse_path):
        ctypes.CDLL(cusparse_path)
except Exception as exc:
    print(f"[Warning] Failed to apply JAX/CUDA fix: {exc}")


def fix_image_shape(x):
    if isinstance(x, np.ndarray):
        x = jnp.array(x)
    shape = x.shape
    if len(shape) == 3 and shape == (128, 128, 3):
        x = jnp.expand_dims(x, axis=0)
    elif len(shape) == 4 and shape == (1, 128, 128, 3):
        pass
    else:
        x = x.reshape((1, 128, 128, 3))
    return x


def split_pkl_paths(dir_path, holdout_count):
    if not os.path.exists(dir_path):
        return [], []
    pkl_paths = sorted(glob.glob(os.path.join(dir_path, "*.pkl")))
    if holdout_count <= 0:
        return pkl_paths, []
    if len(pkl_paths) <= holdout_count:
        return [], pkl_paths
    return pkl_paths[:-holdout_count], pkl_paths[-holdout_count:]


def extract_traj_images(traj, frame_count, take_from="tail"):
    images = []
    if not isinstance(traj, list):
        return images

    if frame_count > 0:
        if take_from == "head":
            selected = traj[:frame_count]
        else:
            selected = traj[-frame_count:]
    else:
        selected = traj

    for transition in selected:
        next_obs = transition.get("next_observations", {})
        img = next_obs.get(IMAGE_KEY)
        if img is None:
            continue
        img = np.asarray(img)
        if img.shape == (1, 128, 128, 3):
            img = img[0]
        if img.shape == (128, 128, 3):
            images.append(img)
    return images


def load_images_from_transition_dir(dir_path, frame_count, take_from="tail", split="holdout"):
    all_images = []
    session_results = []
    if not os.path.exists(dir_path):
        print(f"Transition dir not found, skip: {dir_path}")
        return all_images, session_results

    train_paths, holdout_paths = split_pkl_paths(dir_path, HOLDOUT_TRAJ_COUNT)
    if split == "train":
        selected_paths = train_paths
    elif split == "all":
        selected_paths = train_paths + holdout_paths
    else:
        selected_paths = holdout_paths

    print(
        f"Loading {os.path.basename(dir_path)} with "
        f"train={len(train_paths)} holdout={len(holdout_paths)} using={len(selected_paths)} for {split}"
    )
    for path in tqdm(selected_paths, desc=f"Loading {os.path.basename(dir_path)}"):
        try:
            with open(path, "rb") as handle:
                traj = pkl.load(handle)
            imgs = extract_traj_images(traj, frame_count=frame_count, take_from=take_from)
            if imgs:
                all_images.extend(imgs)
                session_results.append(
                    {
                        "name": os.path.basename(path),
                        "count": len(imgs),
                        "imgs": imgs,
                    }
                )
        except Exception as exc:
            print(f"Error loading trajectory {path}: {exc}")
    return all_images, session_results


def main():
    print("Loading holdout trajectory-backed classifier dataset...")
    success_images, success_results = load_images_from_transition_dir(
        SUCCESS_DIR,
        frame_count=SUCCESS_TAIL_FRAMES_PER_TRAJ,
        take_from="tail",
        split="holdout",
    )
    failure_images, failure_results = load_images_from_transition_dir(
        SUCCESS_DIR,
        frame_count=FAILURE_HEAD_FRAMES_PER_TRAJ,
        take_from="head",
        split="holdout",
    )

    session_results = []
    for item in success_results:
        session_results.append({**item, "label": "SUCCESS"})
    for item in failure_results:
        session_results.append({**item, "label": "FAILURE"})

    total_images = len(success_images) + len(failure_images)
    print(f"\nSummary:")
    print(f" - Success Images: {len(success_images)}")
    print(f" - Failure Images: {len(failure_images)}")
    print(" - Failure Image Source: success trajectories (head frames)")
    print(f" - Success Tail Frames Per Trajectory: {SUCCESS_TAIL_FRAMES_PER_TRAJ}")
    print(f" - Failure Head Frames Per Trajectory: {FAILURE_HEAD_FRAMES_PER_TRAJ}")
    print(f" - Holdout Trajectories Per Class: {HOLDOUT_TRAJ_COUNT}")
    if total_images == 0:
        print("No images found.")
        return

    rng = jax.random.PRNGKey(0)
    init_img = fix_image_shape(success_images[0] if success_images else failure_images[0])
    sample = {"image_0": init_img, "state": jnp.zeros((1, 14))}
    print(f"Loading classifier from {CHECKPOINT_PATH}...")
    classifier_func = load_classifier_func(
        key=rng,
        sample=sample,
        image_keys=["image_0"],
        checkpoint_path=os.path.abspath(CHECKPOINT_PATH),
    )

    total_stats = {"SUCCESS": {"probs": [], "correct": 0}, "FAILURE": {"probs": [], "correct": 0}}
    all_probs = []
    all_labels = []

    print("\n" + "-" * 70)
    print(f"{'Trajectory/File':<40} | {'Label':<8} | {'Count':<6} | {'Correct':<8} | {'Wrong':<6}")
    print("-" * 70)

    for res in session_results:
        session_correct = 0
        session_total = len(res["imgs"])
        expected_label = 1 if res["label"] == "SUCCESS" else 0

        for img in res["imgs"]:
            img_jax = fix_image_shape(img)
            logits = classifier_func({"image_0": img_jax})
            prob = float(nn.sigmoid(logits).item())
            pred_label = 1 if prob > 0.5 else 0
            all_probs.append(prob)
            all_labels.append(expected_label)
            total_stats[res["label"]]["probs"].append(prob)
            if pred_label == expected_label:
                session_correct += 1
                total_stats[res["label"]]["correct"] += 1

        session_wrong = session_total - session_correct
        print(f"{res['name']:<40} | {res['label']:<8} | {session_total:<6} | {session_correct:<8} | {session_wrong:<6}")

    print("-" * 70)

    success_mean = np.mean(total_stats["SUCCESS"]["probs"]) if total_stats["SUCCESS"]["probs"] else 0.0
    failure_mean = np.mean(total_stats["FAILURE"]["probs"]) if total_stats["FAILURE"]["probs"] else 0.0
    total_success = len(total_stats["SUCCESS"]["probs"])
    total_failure = len(total_stats["FAILURE"]["probs"])
    overall_correct = total_stats["SUCCESS"]["correct"] + total_stats["FAILURE"]["correct"]
    overall_total = total_success + total_failure

    print("\n" + "=" * 50)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Success Set ({total_success} images)")
    print(f" - Correct:  {total_stats['SUCCESS']['correct']}")
    print(f" - Wrong:    {total_success - total_stats['SUCCESS']['correct']}")
    print(f" - Avg Prob: {success_mean:.4f}")
    print(f"\nFailure Set ({total_failure} images)")
    print(f" - Correct:  {total_stats['FAILURE']['correct']}")
    print(f" - Wrong:    {total_failure - total_stats['FAILURE']['correct']}")
    print(f" - Avg Prob: {failure_mean:.4f}")
    print("-" * 50)
    print(f"Overall Accuracy: {overall_correct / max(overall_total, 1):.4f} ({overall_correct}/{overall_total})")
    print("=" * 50)

    out_dir = os.path.abspath(CHECKPOINT_PATH)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({"prob": all_probs, "label": all_labels})
    df.to_csv(os.path.join(out_dir, "eval_classifier_probs.csv"), index=False)

    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="prob", hue="label", bins=20, stat="density", common_norm=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_classifier_hist.png"))


if __name__ == "__main__":
    main()
