#!/bin/bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
GPU_ZY_PREFIX=/home/sj/miniconda3/envs/gpu_zy
export LD_LIBRARY_PATH="${GPU_ZY_PREFIX}/lib/python3.12/site-packages/nvidia/cublas/lib:${GPU_ZY_PREFIX}/lib/python3.12/site-packages/nvidia/cudnn/lib:${GPU_ZY_PREFIX}/lib/python3.12/site-packages/nvidia/cufft/lib:${GPU_ZY_PREFIX}/lib/python3.12/site-packages/nvidia/cusolver/lib:${GPU_ZY_PREFIX}/lib/python3.12/site-packages/nvidia/cusparse/lib:${GPU_ZY_PREFIX}/lib/python3.12/site-packages/nvidia/nccl/lib:${GPU_ZY_PREFIX}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${GPU_ZY_PREFIX}/lib/python3.12/site-packages/nvidia --xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0"

echo "[BC Eval GPU] Evaluating on real env/server path (GPU)"

conda run -n gpu_zy env \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" \
  TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR}" \
  XLA_FLAGS="${XLA_FLAGS}" \
  python train_pick_place_bc_gpu.py \
  --mode=eval \
  --render
