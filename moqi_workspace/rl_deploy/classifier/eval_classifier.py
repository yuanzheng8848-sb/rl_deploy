import jax
# Set XLA to not preallocate all memory
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from jax import numpy as jnp
import flax.linen as nn
import numpy as np
import glob
import cv2
import sys
import ctypes
import site
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import pandas as pd
import seaborn as sns

from serl_launcher.networks.reward_classifier import load_classifier_func

# --- JAX/CUDA Fix ---
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
    print(f"[Warning] Failed to apply JAX/CUDA fix: {e}")

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

# 与 train 一致：demo/record_data，每 session 的 cam_2 前 10 帧=失败，后 10 帧=成功
RECORD_DATA_DIR = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/demo/record_data"
CAM_SUBDIR = "cam_2_rgb"
N_FAILURE_FRAMES = 10
N_SUCCESS_FRAMES = 10
EXTRA_FAILURE_DIR = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/extra_failure_images"


def load_session_first_last(session_path):
    """从 session 的 cam_2_rgb 加载前 N 帧(失败)与后 N 帧(成功)。返回 (failure_imgs, success_imgs)。"""
    img_dir = os.path.join(session_path, "images", CAM_SUBDIR)
    if not os.path.exists(img_dir):
        return [], []
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    total = len(image_paths)
    if total < N_FAILURE_FRAMES + N_SUCCESS_FRAMES:
        return [], []
    n_fail = min(N_FAILURE_FRAMES, total)
    n_succ = min(N_SUCCESS_FRAMES, total)
    failure_paths = image_paths[:n_fail]
    success_paths = image_paths[-n_succ:]
    failure_imgs = []
    success_imgs = []
    for path in failure_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            failure_imgs.append(img)
        except Exception:
            pass
    for path in success_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            success_imgs.append(img)
        except Exception:
            pass
    return failure_imgs, success_imgs

def load_extra_failure_images(extra_failure_dir):
    """加载训练过程手动采集的额外失败图像。返回 HxWx3 RGB 列表。"""
    images = []
    if not os.path.exists(extra_failure_dir):
        print(f"Extra failure image dir not found, skip: {extra_failure_dir}")
        return images
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(extra_failure_dir, p)))
    paths = sorted(paths)
    print(f"Found {len(paths)} extra failure images in {extra_failure_dir}")
    for path in paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            images.append(img)
        except Exception:
            pass
    return images

def main():
    CHECKPOINT_PATH = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/classifier_ckpt"

    print("Loading data...")
    print(f"Data dir: {RECORD_DATA_DIR}, cam: {CAM_SUBDIR}, first {N_FAILURE_FRAMES}=FAILURE, last {N_SUCCESS_FRAMES}=SUCCESS")
    success_images = []
    failure_images = []
    session_results = []

    if os.path.exists(RECORD_DATA_DIR):
        sessions = sorted([d for d in os.listdir(RECORD_DATA_DIR) if os.path.isdir(os.path.join(RECORD_DATA_DIR, d)) and d.startswith("session_")])
        for session in tqdm(sessions, desc="Loading Sessions"):
            session_path = os.path.join(RECORD_DATA_DIR, session)
            fail_imgs, succ_imgs = load_session_first_last(session_path)
            if not fail_imgs and not succ_imgs:
                continue
            success_images.extend(succ_imgs)
            failure_images.extend(fail_imgs)
            if fail_imgs:
                session_results.append({"name": session, "count": len(fail_imgs), "label": "FAILURE", "imgs": fail_imgs})
            if succ_imgs:
                session_results.append({"name": session, "count": len(succ_imgs), "label": "SUCCESS", "imgs": succ_imgs})

    extra_failure_images = load_extra_failure_images(EXTRA_FAILURE_DIR)
    if extra_failure_images:
        failure_images.extend(extra_failure_images)
        session_results.append(
            {
                "name": "extra_failure_images",
                "count": len(extra_failure_images),
                "label": "FAILURE",
                "imgs": extra_failure_images,
            }
        )
    
    total_images = len(success_images) + len(failure_images)
    print(f"\nData Loading Complete:")
    print(f" - Total Success Images: {len(success_images)}")
    print(f" - Total Failure Images: {len(failure_images)}")
    print(f" - Extra Failure Images: {len(extra_failure_images)}")
    
    if total_images == 0:
        print("No images found.")
        return

    # 3. Load Classifier
    rng = jax.random.PRNGKey(0)
    image_keys = ["image_0"]
    
    # Init sample
    init_img = fix_image_shape(success_images[0] if success_images else failure_images[0])
    sample = {"image_0": init_img, "state": jnp.zeros((1, 14))}
    
    print(f"Loading classifier from {CHECKPOINT_PATH}...")
    try:
        classifier_func = load_classifier_func(
            key=rng,
            sample=sample,
            image_keys=image_keys,
            checkpoint_path=os.path.abspath(CHECKPOINT_PATH),
        )
    except Exception as e:
        print(f"Failed to load classifier: {e}")
        return

    # 4. Evaluate
    print("\nStarting Evaluation...")
    total_stats = {"SUCCESS": {"probs": [], "correct": 0}, "FAILURE": {"probs": [], "correct": 0}}
    
    print("\n" + "-"*60)
    print(f"{'Session Name':<30} | {'Label':<8} | {'Count':<6} | {'Correct':<8} | {'Wrong':<6}")
    print("-" * 60)

    for res in session_results:
        session_correct = 0
        session_total = len(res["imgs"])
        expected_label = 1 if res["label"] == "SUCCESS" else 0
        
        if session_total == 0:
            print(f"{res['name']:<30} | {res['label']:<8} | {session_total:<6} | {'N/A':<8} | {'N/A':<6}")
            continue

        for img in res["imgs"]:
            img_jax = fix_image_shape(img)
            obs = {"image_0": img_jax}
            logits = classifier_func(obs)
            prob = nn.sigmoid(logits).item()
            
            total_stats[res["label"]]["probs"].append(prob)
            pred_label = 1 if prob > 0.5 else 0
            if pred_label == expected_label:
                session_correct += 1
                total_stats[res["label"]]["correct"] += 1
        
        session_wrong = session_total - session_correct
        print(f"{res['name']:<30} | {res['label']:<8} | {session_total:<6} | {session_correct:<8} | {session_wrong:<6}")

    print("-" * 60)

    # Mean prob (no std)
    def get_mean(probs):
        if not probs:
            return 0.0
        return np.mean(probs)

    success_mean = get_mean(total_stats["SUCCESS"]["probs"])
    failure_mean = get_mean(total_stats["FAILURE"]["probs"])

    total_success = len(total_stats["SUCCESS"]["probs"])
    total_failure = len(total_stats["FAILURE"]["probs"])
    overall_correct = total_stats["SUCCESS"]["correct"] + total_stats["FAILURE"]["correct"]
    overall_total = total_success + total_failure
    overall_wrong = overall_total - overall_correct
    success_wrong = total_success - total_stats["SUCCESS"]["correct"]
    failure_wrong = total_failure - total_stats["FAILURE"]["correct"]

    print("\n" + "="*50)
    print("FINAL EVALUATION SUMMARY")
    print("="*50)
    print(f"Success Set ({total_success} images):")
    print(f" - Correct:    {total_stats['SUCCESS']['correct']}")
    print(f" - Wrong:      {success_wrong}")
    print(f" - Avg Prob:   {success_mean:.4f}")

    print(f"\nFailure Set ({total_failure} images):")
    print(f" - Correct:    {total_stats['FAILURE']['correct']}")
    print(f" - Wrong:      {failure_wrong}")
    print(f" - Avg Prob:   {failure_mean:.4f}")

    print("-" * 50)
    print(f"Overall Accuracy: {overall_correct/(overall_total or 1):.4f} ({overall_correct}/{overall_total})")
    print("="*50 + "\n")

    # --- Generate Visualizations ---
    log_dir = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy/classifier/training_logs"
    debug_dir = os.path.join(log_dir, "debug_images")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 1. Confidence Distribution (Histogram/Density)
    plt.figure(figsize=(10, 6))
    if total_stats["SUCCESS"]["probs"]:
        sns.kdeplot(total_stats["SUCCESS"]["probs"], fill=True, label="Success Images", color="tab:blue")
    if total_stats["FAILURE"]["probs"]:
        sns.kdeplot(total_stats["FAILURE"]["probs"], fill=True, label="Failure Images", color="tab:red")
    plt.axvline(0.5, color="black", linestyle="--", label="Threshold (0.5)")
    plt.xlabel("Confidence (Probability)")
    plt.ylabel("Density")
    plt.title("Classifier Confidence Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "confidence_distribution.png"))
    plt.close()

    # 2. Confusion Matrix (Manual)
    tp = total_stats["SUCCESS"]["correct"]
    fn = total_success - tp
    tn = total_stats["FAILURE"]["correct"]
    fp = total_failure - tn
    
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Failure', 'Pred Success'], 
                yticklabels=['Actual Failure', 'Actual Success'])
    plt.title('Confusion Matrix (Threshold=0.5)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
    plt.close()

    # 3. ROC and PR Curves (Manual)
    all_probs = np.array(total_stats["SUCCESS"]["probs"] + total_stats["FAILURE"]["probs"])
    all_labels = np.array([1]*len(total_stats["SUCCESS"]["probs"]) + [0]*len(total_stats["FAILURE"]["probs"]))
    
    thresholds = np.linspace(0, 1, 101)
    tprs, fprs = [], []
    precisions, recalls = [], []
    
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        tp_thr = np.sum((preds == 1) & (all_labels == 1))
        fp_thr = np.sum((preds == 1) & (all_labels == 0))
        tn_thr = np.sum((preds == 0) & (all_labels == 0))
        fn_thr = np.sum((preds == 0) & (all_labels == 1))
        
        tpr = tp_thr / (tp_thr + fn_thr) if (tp_thr + fn_thr) > 0 else 0
        fpr = fp_thr / (fp_thr + tn_thr) if (fp_thr + tn_thr) > 0 else 0
        prec = tp_thr / (tp_thr + fp_thr) if (tp_thr + fp_thr) > 0 else 1
        rec = tpr
        
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(prec)
        recalls.append(rec)

    # Calculate AUC (Trapezoidal rule)
    auc = np.abs(np.trapz(tprs, fprs))
    
    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "roc_curve.png"))
    plt.close()
    
    # Plot PR
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "precision_recall_curve.png"))
    plt.close()

    # 4. Session Error Analysis & Debug Images
    session_names = [r["name"] for r in session_results if len(r["imgs"]) > 0]
    session_errors = []
    
    # Clean debug images first to avoid old results
    for f in glob.glob(os.path.join(debug_dir, "*.jpg")):
        os.remove(f)

    for r in session_results:
        if len(r["imgs"]) > 0:
            errors = 0
            label_int = 1 if r["label"] == "SUCCESS" else 0
            label_str = r["label"].lower()
            
            for i, img in enumerate(r["imgs"]):
                img_jax = fix_image_shape(img)
                logits = classifier_func({"image_0": img_jax})
                prob = nn.sigmoid(logits).item()
                
                is_correct = (1 if prob > 0.5 else 0) == label_int
                if not is_correct:
                    errors += 1
                    # Save debug image
                    filename = f"{label_str}_{prob:.4f}_{r['name']}_{i}.jpg"
                    save_path = os.path.join(debug_dir, filename)
                    # Convert RGB to BGR for cv2 saving
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, img_bgr)
            
            session_errors.append(errors)
    
    if session_names:
        err_df = pd.DataFrame({"Session": session_names, "Errors": session_errors})
        err_df = err_df.sort_values("Errors", ascending=False)
        
        plt.figure(figsize=(12, 8))
        plt.barh(err_df["Session"], err_df["Errors"], color="tab:red")
        plt.xlabel("Number of Misclassified Images")
        plt.title("Per-Session Error Analysis (Threshold=0.5)")
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "session_error_analysis.png"))
        plt.close()

    print(f"Advanced evaluation charts saved to: {log_dir}")

if __name__ == "__main__":
    main()
