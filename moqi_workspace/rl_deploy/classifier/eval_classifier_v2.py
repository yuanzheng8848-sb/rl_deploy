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

def load_images_from_session(session_path):
    img_dir = os.path.join(session_path, "images", "cam_1_rgb")
    if not os.path.exists(img_dir):
        return []
    
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) 
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            images.append(img)
        except Exception:
            pass
    return images

def main():
    RECORD_DATA_DIR = "/home/peter/Desktop/zy/moqi_workspace/record_data"
    EXTRA_FAILURE_DIR = "/home/peter/Desktop/zy/moqi_workspace/rl_deploy/classifier/recording_20260122_203118"
    CHECKPOINT_PATH = "classifier/classifier_ckpt_cam1_v2" # Relative path in current dir

    print("Loading data...")
    success_images = []
    failure_images = []
    
    # 1. Load from Record Data
    if os.path.exists(RECORD_DATA_DIR):
        sessions = sorted([d for d in os.listdir(RECORD_DATA_DIR) if os.path.isdir(os.path.join(RECORD_DATA_DIR, d))])
        for session in tqdm(sessions, desc="Loading Sessions"):
            if session in ["success", "failure"]: continue
            
            session_path = os.path.join(RECORD_DATA_DIR, session)
            imgs = load_images_from_session(session_path)
            
            if len(imgs) < 20: continue
            
            failure_images.extend(imgs[:30])
            success_images.extend(imgs[-30:])
    
    print(f"Loaded {len(success_images)} success and {len(failure_images)} failure images from sessions.")

    # 2. Load Extra Failure Images
    extra_failure_images = []
    if os.path.exists(EXTRA_FAILURE_DIR):
        print(f"Loading extra failure images from {EXTRA_FAILURE_DIR}...")
        extra_paths = sorted(glob.glob(os.path.join(EXTRA_FAILURE_DIR, "*.jpg")))
        for path in tqdm(extra_paths, desc="Loading Extra"):
            try:
                img = cv2.imread(path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                extra_failure_images.append(img)
            except Exception:
                pass
    
    print(f"Loaded {len(extra_failure_images)} extra failure images.")
    
    # Combine failure images for evaluation (but keep separate for reporting)
    all_failure_images = failure_images + extra_failure_images
    
    if not success_images and not all_failure_images:
        print("No images found.")
        return

    # 3. Load Classifier
    rng = jax.random.PRNGKey(0)
    image_keys = ["image_0"]
    
    # Init sample
    init_img = fix_image_shape(success_images[0] if success_images else all_failure_images[0])
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
    def evaluate_set(images, label_name, expected_label):
        correct = 0
        total = len(images)
        if total == 0: return 0.0
        
        print(f"\nEvaluating {label_name} ({total} images)...")
        for img in tqdm(images):
            img_jax = fix_image_shape(img)
            obs = {"image_0": img_jax}
            logits = classifier_func(obs)
            prob = nn.sigmoid(logits).item()
            
            pred_label = 1 if prob > 0.5 else 0
            if pred_label == expected_label:
                correct += 1
                
        acc = correct / total
        print(f"{label_name} Accuracy: {acc:.4f} ({correct}/{total})")
        return acc

    acc_success = evaluate_set(success_images, "Original Success", 1)
    acc_failure_orig = evaluate_set(failure_images, "Original Failure", 0)
    acc_failure_extra = evaluate_set(extra_failure_images, "Extra Failure", 0)
    
    # Overall
    total_images = len(success_images) + len(failure_images) + len(extra_failure_images)
    total_correct = (acc_success * len(success_images)) + \
                    (acc_failure_orig * len(failure_images)) + \
                    (acc_failure_extra * len(extra_failure_images))
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Original Success Accuracy: {acc_success:.4f}")
    print(f"Original Failure Accuracy: {acc_failure_orig:.4f}")
    print(f"Extra Failure Accuracy:    {acc_failure_extra:.4f}")
    print("-" * 40)
    print(f"Overall Accuracy:          {total_correct/total_images:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
