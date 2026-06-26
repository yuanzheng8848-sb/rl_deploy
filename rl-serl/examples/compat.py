"""Runtime compatibility shims for rl-serl.

Import this module FIRST (before jax / flax) in every entrypoint. It:
  1. Wires up sys.path so that rl-serl can import its own packages AND the
     reused upstream code (no `pip install -e` required):
       - rl-serl/rl_launcher               (forwarding layer over serl_launcher)
       - rl-serl/rl_robot_infra            (OpenArm env / wrappers / server)
       - hil-serl/serl_launcher            (SAC/RLPD/encoder/buffer/classifier networks)
       - hil-serl/serl_robot_infra         (transformations, reward classifier wrapper)
       - moqi_workspace/pyroki             (realsense_camera, robot_ik_solver, viser_base)
       - moqi_workspace/openarm            (openarm_controller_2)
     NOTE: moqi_workspace/rl_deploy is intentionally NOT added — rl-serl must not
     depend on rl_deploy (it will be deleted).
  2. Configures JAX CUDA library paths before JAX touches the GPU.
  3. Applies monkey-patches that keep older HIL-SERL/Flax code working on newer JAX.

Merged from rl_deploy/train.py (lines 26-82) and rl_deploy/classifier/eval_classifier.py.
"""
import ctypes
import os
import site
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. sys.path wiring
# ---------------------------------------------------------------------------
# rl-serl/examples/compat.py -> parents[1] == rl-serl, parents[2] == zy (repo root)
RL_SERL_ROOT = Path(__file__).resolve().parents[1]
ZY_ROOT = RL_SERL_ROOT.parent

_REUSE_PATHS = [
    RL_SERL_ROOT / "rl_launcher",       # rl-serl own packages (no pip install needed)
    RL_SERL_ROOT / "rl_robot_infra",
    ZY_ROOT / "hil-serl" / "serl_launcher",
    ZY_ROOT / "hil-serl" / "serl_robot_infra",
    ZY_ROOT / "moqi_workspace" / "pyroki",
    ZY_ROOT / "moqi_workspace" / "openarm",
]

for _p in _REUSE_PATHS:
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)


# ---------------------------------------------------------------------------
# 2. CUDA library path configuration (must run before any JAX GPU work)
# ---------------------------------------------------------------------------
def _configure_cuda_paths():
    try:
        nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
    except Exception:
        return
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

    try:
        nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
        cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
        if os.path.exists(nvjitlink_path):
            ctypes.CDLL(nvjitlink_path)
        if os.path.exists(cusparse_path):
            ctypes.CDLL(cusparse_path)
    except Exception as exc:
        print(f"[compat] Failed to preload CUDA libraries: {exc}")


_configure_cuda_paths()


# ---------------------------------------------------------------------------
# 3. JAX monkey-patches for newer JAX releases used with older HIL-SERL code
# ---------------------------------------------------------------------------
def _patch_jax():
    import jax

    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map
    if not hasattr(jax, "tree_leaves"):
        jax.tree_leaves = jax.tree_util.tree_leaves

    _orig_shaped_array_update = jax.core.ShapedArray.update

    def _compat_shaped_array_update(self, *args, **kwargs):
        # `named_shape` was removed from newer JAX ShapedArray constructors, but
        # some upstream Flax/JAX interactions may still pass it around.
        kwargs.pop("named_shape", None)
        return _orig_shaped_array_update(self, *args, **kwargs)

    jax.core.ShapedArray.update = _compat_shaped_array_update


_patch_jax()
