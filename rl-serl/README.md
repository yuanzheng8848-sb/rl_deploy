# rl-serl

OpenArm 双臂操作的 HIL-SERL 训练代码，按照 [hil-serl](https://github.com/rail-berkeley/serl) 三包架构重组。

## 架构

仿照 hil-serl 拆分为三个包：

```
rl-serl/
├── rl_launcher/              # 算法层（转发 serl_launcher，不重写 SAC/RLPD）
│   └── rl_launcher/          # agents, networks, data, utils, wrappers
├── rl_robot_infra/           # 硬件层（OpenArm gym env + wrappers + server）
│   ├── openarm_env/
│   │   ├── envs/             # openarm_env.py, local_openarm_env.py, wrappers.py
│   │   ├── camera/           # local_camera.py (三相机设备常量 + 构造)
│   │   └── mock_hardware.py
│   └── robot_servers/
│       └── openarm_server.py # Flask server（/pose, /servo/*）
└── examples/                 # 训练入口 + 任务配置
    ├── compat.py             # sys.path 注入 + CUDA/JAX patch（必须首行 import）
    ├── train_rlpd.py         # actor / learner / eval 入口
    ├── record_demos.py       # 双 SpaceMouse 录制 demo（ENTER=success, SPACE=failure）
    ├── train_reward_classifier.py  # 训练视觉奖励分类器
    ├── eval_reward_classifier.py   # 评估分类器
    ├── run_learner.sh / run_actor.sh / run_eval.sh
    └── experiments/
        ├── config.py         # DefaultTrainingConfig 基类
        ├── mappings.py       # exp_name → TrainConfig 注册表
        └── openarm_pickplace/
            └── config.py     # 任务配置（env 组装 + classifier 奖励对接）
```

## 复用关系

- **算法**：完全复用 `hil-serl/serl_launcher`（SAC/RLPD/encoder/buffer），`rl_launcher` 只做转发层
- **硬件底层**：复用 `moqi_workspace/{pyroki, openarm, IK}`（realsense、IK solver、openarm_controller_2）
- **奖励分类器 wrapper**：复用 `hil-serl/serl_robot_infra` 的 `MultiCameraBinaryRewardClassifierWrapper`
- **迁移自 rl_deploy**：OpenArm 特定的 env / wrappers / server / 训练入口

## 环境准备

```bash
conda activate zy  # 或你的环境（需有 jax + flax）
cd rl-serl
# 方式 1：安装包
pip install -e rl_launcher
pip install -e rl_robot_infra
# 方式 2：设 PYTHONPATH（examples 下脚本已自动设置）
export PYTHONPATH=$PWD/rl_robot_infra:$PWD/rl_launcher:$PWD/examples
```

## 训练流程

### 1. 录制 demo

```bash
cd examples
python record_demos.py --exp_name openarm_pickplace
# 用 SpaceMouse 操作，ENTER 保存成功轨迹，SPACE 保存失败轨迹，ESC 退出
# 保存到 examples/experiments/openarm_pickplace/demo/collected/{success,failure}/
# 原始图像保存到 examples/experiments/openarm_pickplace/demo/collected/raw_images/
```

### 2. 训练奖励分类器

```bash
python train_reward_classifier.py --exp_name openarm_pickplace --num_epochs 100
# 从 success 轨迹的 tail 帧取正样本、head 帧取负样本
# 使用 demo 中已由 NetworkPrimaryImageCropWrapper 处理过的 image_primary，不做二次裁剪
# 默认读取 examples/experiments/openarm_pickplace/demo/collected/success/
# 保存到 examples/experiments/openarm_pickplace/classifier_ckpt/
```

### 3. RLPD 训练（learner + actor）

在两个终端同时运行：

```bash
# Terminal 1: learner（fake env，纯更新网络）
cd examples
bash run_learner.sh openarm_pickplace ./checkpoints_rlpd

# Terminal 2: actor（真实环境 + SpaceMouse 干预 + classifier 奖励）
bash run_actor.sh openarm_pickplace ./checkpoints_rlpd localhost
```

### 4. 评估

```bash
bash run_eval.sh openarm_pickplace ./checkpoints_rlpd 10 0
# 参数：exp_name, checkpoint_path, n_trajs, checkpoint_step(0=latest)
```

## 奖励设计（关键）

与原 rl_deploy 的主要区别：

- **奖励来源**：视觉分类器（`MultiCameraBinaryRewardClassifierWrapper`），不再用键盘手动标
- **输入相机**：**单相机** `image_primary`（顶部 USB 相机），不再用三相机拼接
- **裁剪一致性**：主相机只由 `NetworkPrimaryImageCropWrapper` 裁剪一次（crop_ratio=0.3, y_offset=0）；demo pkl、分类器、critic/policy 都消费这个已裁剪的 `image_primary`
- **wrapper 顺序**：`NetworkPrimaryImageCropWrapper` → `SERLObsWrapper` → `MultiCameraBinaryRewardClassifierWrapper`，确保分类器读到已裁剪的 `image_primary`

见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md) §4 完整设计。

## 已剥离的功能（相比 rl_deploy）

1. **Arm-Focus 单臂聚焦**：DualSpacemouseIntervention 中 left/right 恒 active，动作不 mask
2. **KeyboardRewardWrapper**：奖励改由 classifier 提供
3. **Handoff-Focus 标记 + demo buffer 重复插入**
4. **Episode 诊断指标**（joint distance / monotonicity / autonomy / ...）

这些是原 rl_deploy 为早期探索保留的脚手架，现已不需要。

## 待验证项（见 REFACTOR_PLAN.md 进度表）

- [ ] train_rlpd 端到端冒烟（learner --mock 跑通是首要）
- [ ] classifier 奖励链路用真实 ckpt 验证
- [ ] SpaceMouse /servo 闭环在真机验证

## 参考

- 上游：[hil-serl](https://github.com/rail-berkeley/serl)
- 迁移计划：[REFACTOR_PLAN.md](REFACTOR_PLAN.md)
