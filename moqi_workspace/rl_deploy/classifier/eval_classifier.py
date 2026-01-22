import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import os
import glob
import cv2
import sys
import ctypes
import site
from serl_launcher.networks.reward_classifier import load_classifier_func

# --- 强制指定 JAX 使用的 NVIDIA 库路径 ---
# 必须在导入 jax 或其他可能使用它的库之前完成此操作
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

    # 设置 XLA_FLAGS 以帮助 JAX 找到 CUDA 数据目录
    os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={nvidia_base}"

    # 显式按依赖顺序预加载库
    nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
    if os.path.exists(nvjitlink_path):
        ctypes.CDLL(nvjitlink_path)
    
    cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
    if os.path.exists(cusparse_path):
        ctypes.CDLL(cusparse_path)
        
except Exception as e:
    print(f"[Warning] Failed to apply JAX/CUDA fix: {e}")

def fix_image_shape(x):
    """
    Ensure image shape is (1, 128, 128, 3) for the classifier query
    """
    # If numpy array, convert to jnp
    if isinstance(x, np.ndarray):
        x = jnp.array(x)
        
    shape = x.shape
    if len(shape) == 3 and shape[0] == 128 and shape[1] == 128 and shape[2] == 3:
        x = jnp.expand_dims(x, axis=0)
    elif len(shape) == 4 and shape[0] == 1 and shape[1] == 128 and shape[2] == 128 and shape[3] == 3:
        pass # Already correct
    else:
        # Try to clean up
         x = x.reshape((1, 128, 128, 3))
    return x
def main():
    # Target Directory (User specified)
    TARGET_IMG_DIR = "/home/sj/Desktop/zy/moqi_workspace/record_data11/success/session_0004_30hz_20251205_121317/images/cam_1_rgb"
    
    # Checkpoint Path (User specified)
    # Note: Corrected path based on previous interactions or user request
    CHECKPOINT_PATH = "/home/sj/Desktop/zy/moqi_workspace/rl_deploy_2/classifier_ckpt_cam1"

    if not os.path.exists(TARGET_IMG_DIR):
        raise FileNotFoundError(f"Image dir not found: {TARGET_IMG_DIR}")

    print(f"Loading images from: {TARGET_IMG_DIR}")
    
    image_paths = sorted(glob.glob(os.path.join(TARGET_IMG_DIR, "*.jpg")))
    image_paths += sorted(glob.glob(os.path.join(TARGET_IMG_DIR, "*.png")))
    
    if not image_paths:
        print("No images found!")
        return

    print(f"Found {len(image_paths)} images.")

    # 1. Load Images
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    # 2. Load Classifier
    rng = jax.random.PRNGKey(0)
    image_keys = ["image_0"]
    
    # Init sample
    if not images:
        print("No valid images loaded.")
        return

    init_img = fix_image_shape(images[0])
    sample = {"image_0": init_img, "state": jnp.zeros((1, 14))}
    
    print(f"Loading classifier from {CHECKPOINT_PATH}...")
    classifier_func = load_classifier_func(
        key=rng,
        sample=sample,
        image_keys=image_keys,
        checkpoint_path=CHECKPOINT_PATH,
    )
    
    # 3. Evaluate
    print("\n--- Evaluation Results ---")
    print(f"{'Frame':<10} | {'Prob':<10} | {'Bar'}")
    print("-" * 40)
    
    for i, (img_np, path) in enumerate(zip(images, valid_paths)):
        img_jax = fix_image_shape(img_np)
        obs = {"image_0": img_jax}
        logits = classifier_func(obs)
        prob = nn.sigmoid(logits).item()
        
        # Simple ascii bar
        bar_len = int(prob * 20)
        bar = "#" * bar_len
        
        filename = os.path.basename(path)
        print(f"{filename:<10} | {prob:.4f}     | {bar}")

    print("-" * 40)
    print("Done.")

if __name__ == "__main__":
    main()
