# rl-serl BC Warmup 使用指南

## 概述

BC (Behavioral Cloning) warmup 为 RLPD 训练提供预训练的策略初始化。完整工作流：

```
录制 demo → 训练 classifier → 训练 BC → 离线评估 → 真机评估 → RLPD 训练
```

## 目录结构

所有任务产物统一存放在 `examples/experiments/<exp_name>/` 下：

```
examples/experiments/<exp_name>/
├── demo/
│   ├── collected/
│   │   ├── success/              # BC 训练数据源
│   │   └── failure/              # classifier 负样本
│   └── raw_images/
│       ├── success/
│       └── failure/
├── classifier_ckpt/              # reward classifier checkpoint
├── checkpoints_bc/               # BC warmup checkpoint
├── bc_eval/
│   ├── offline/                  # 离线评估结果
│   └── real/                     # 真机评估结果
└── checkpoints_rlpd/             # RLPD checkpoint 和 buffer
    ├── buffer/                   # replay buffer
    └── demo_buffer/              # online intervention demo
```

## 完整工作流

### 0. 启动 OpenArm Hardware Server

录制 demo、真机评估 BC、RLPD actor 都需要先启动 OpenArm hardware server。推荐从 `rl_robot_infra` 目录用模块方式启动，确保 `openarm_env` 能被 Python 正确导入：

```bash
conda activate zy
cd /home/sj/Desktop/zy/rl-serl/rl_robot_infra
python -m robot_servers.openarm_server
```

启动成功后应看到类似日志：

```text
Starting OpenArm Server on port 5000...
```

**注意：**
- Server 默认监听 `http://127.0.0.1:5000/`，与 `DefaultOpenArmConfig.SERVER_URL` 一致
- 不要从 `examples/` 目录直接运行 `python /home/sj/Desktop/zy/rl-serl/rl_robot_infra/robot_servers/openarm_server.py`，否则可能出现 `ModuleNotFoundError: No module named 'openarm_env'`
- 如果要使用真硬件，请确认 `robot_servers/openarm_server.py` 中 `USE_MOCK = False`；mock 调试则保持 `USE_MOCK = True`

### 1. 录制 Demo

```bash
cd rl-serl/examples
python record_demos.py --exp_name openarm_pickplace
```

录制前请在另一个终端保持 OpenArm hardware server 运行。

**操作说明：**
- `ENTER` 键：保存为 success demo
- `SPACE` 键：保存为 failure demo  
- `ESC` 键：退出

**建议数量：**
- Success demos: 10-20 条（用于 BC 训练和 RLPD offline demos）
- Failure demos: 5-10 条（用于 classifier 负样本）

**输出位置：**
- `demo/collected/success/*.pkl`
- `demo/collected/failure/*.pkl`
- `demo/collected/raw_images/success/*.png`
- `demo/collected/raw_images/failure/*.png`

### 2. 训练 Reward Classifier

```bash
# 训练
python train_reward_classifier.py \
    --exp_name openarm_pickplace \
    --num_epochs 100

# 评估
python eval_reward_classifier.py \
    --exp_name openarm_pickplace 
```

**训练参数：**
- `--exp_name`: 任务名称
- `--num_epochs`: 训练轮数（默认 100）
- `--batch_size`: batch size（默认 32）
- `--learning_rate`: 学习率（默认 3e-4）

**训练输出：**
- `classifier_ckpt/checkpoint_*`: 保存的 checkpoint
- 终端打印训练 loss 和准确率

**评估参数：**
- `--exp_name`: 任务名称
- `--split`: 评估集（`train` 或 `holdout`，默认 `holdout`）

**评估输出：**
- 终端打印：整体准确率、success/failure 分类准确率、概率统计、误分类样本数
- `classifier_ckpt/eval_probability_distribution_{split}.png`: 概率分布直方图和箱线图
- `classifier_ckpt/misclassified_samples_{split}/`: 前20个误分类样本图像

### 3. 训练 BC Warmup Checkpoint

```bash
python train_bc.py \
    --exp_name openarm_pickplace \
    --train_steps 20000
```

**关键参数：**
- `--exp_name`: 任务名称
- `--train_steps`: 训练步数（默认 20000）
- `--batch_size`: 使用任务配置的 batch_size
- `--checkpoint_period`: checkpoint 保存间隔（默认 5000）
- `--bc_checkpoint_path`: BC checkpoint 目录（默认自动）
- `--success_dir`: success demo 目录（默认自动）
- `--debug`: debug 模式

**输出：**
- `checkpoints_bc/checkpoint_<step>`: BC checkpoint
- `checkpoints_bc/training_log.csv`: 训练日志
  - actor BC loss
  - continuous action MSE/MAE
  - gripper CE loss
  - gripper accuracy
  - joint gripper accuracy

### 4. 离线评估 BC Checkpoint

不连接真机，逐帧预测动作并与 demo 对比，包含 3D 动作轨迹可视化。

```bash
python eval_bc_offline.py \
    --exp_name openarm_pickplace
```

**关键参数：**
- `--exp_name`: 任务名称
- `--bc_checkpoint_path`: BC checkpoint 目录（默认自动）
- `--checkpoint_step`: checkpoint 步数（0 表示 latest）
- `--success_dir`: 评估用 success demo 目录（默认自动）
- `--output_dir`: 输出目录（默认 `bc_eval/offline/`）
- `--max_trajs`: 最大评估轨迹数（0 表示全量）
- `--high_error_count`: 导出的高误差帧数量（默认 50）
- `--seed`: 评估随机种子（默认 42）
- `--argmax` / `--noargmax`: 是否使用 deterministic action mode（默认 `--argmax`）
- `--generate_plots` / `--nogenerate_plots`: 是否生成 PDF 可视化报告（默认 `--generate_plots`）
- `--plot_3d_trajectories` / `--noplot_3d_trajectories`: 是否生成 3D 末端轨迹图（默认 `--plot_3d_trajectories`）
- `--traj_3d_count`: 生成 3D 图的轨迹数（0 表示全部，按 MSE 从高到低）

**常用变体：**
```bash
# 只评估少量 demo，快速检查 checkpoint
python eval_bc_offline.py --exp_name openarm_pickplace --max_trajs 1

# 跳过 PDF，只输出 CSV/JSON
python eval_bc_offline.py --exp_name openarm_pickplace --nogenerate_plots

# 只生成前 5 条轨迹的 3D 图（MSE 最差的 5 条）
python eval_bc_offline.py --exp_name openarm_pickplace --traj_3d_count 5

# 跳过 3D 轨迹图，只要基础报告
python eval_bc_offline.py --exp_name openarm_pickplace --noplot_3d_trajectories
```

**评估指标：**

- **Continuous action MSE/MAE**（12 维连续动作误差）
- **Action cosine similarity**（动作方向一致性）
- **Action magnitude error**（动作幅值误差）
- **Gripper single accuracy**（左右夹爪分类准确率）
- **Gripper joint accuracy**（9 类 joint action 准确率）
- **Temporal smoothness**（时序平滑度，检查抖动）

**输出：**
- `bc_eval/offline/summary.json`: 汇总指标
- `bc_eval/offline/per_trajectory.csv`: 每条轨迹指标
- `bc_eval/offline/per_frame.csv`: 每帧详细数据
- `bc_eval/offline/high_error_frames/`: 高误差帧可视化（PNG）
- `bc_eval/offline/trajectory_3d/*_ee_path.png`: 每条轨迹的末端执行器路径图（仅 `--plot_3d_trajectories`）
- `bc_eval/offline/evaluation_report.pdf`: **可视化报告**（仅 `--generate_plots`）
  - 第1页：误差分布分析
  - 第2页：时间序列分析
  - 第3页：轨迹性能对比
  - 第4页：夹爪分类性能
  - 第5页：指标相关性分析
  - 第6页：评估总结
  - 第7页起：**双臂末端执行器 3D 路径**（仅 `--plot_3d_trajectories`，每条轨迹一页）

**3D 末端轨迹图说明：**

每条轨迹生成一张 1x2 布局的图，显示左右臂末端执行器在笛卡尔空间中的真实运动路径：

- **蓝色曲线**：从 `tcp_pose`（观测中的末端位姿，相对 reset 的米制坐标）提取的真实末端执行器轨迹
- **绿色圆点**：轨迹起点
- **红色星号**：轨迹终点

这是 demo 执行时末端实际走过的空间路径，用于直观检查任务轨迹的形状、幅度和平滑度。

### 5. 真机评估 BC Checkpoint

在真机上直接运行 BC policy，不启动 RLPD。运行前需要先启动 OpenArm hardware server。

```bash
python eval_bc_real.py \
    --exp_name openarm_pickplace \
    --eval_n_trajs 10 \
    --argmax
```

**关键参数：**
- `--exp_name`: 任务名称
- `--bc_checkpoint_path`: BC checkpoint 目录（默认自动）
- `--checkpoint_step`: checkpoint 步数（0 表示 latest）
- `--eval_n_trajs`: 评估轨迹数（默认 10）
- `--argmax`: 是否使用 argmax 采样（默认 True）
- `--save_video`: 是否保存视频
- `--classifier`: 是否使用 classifier（默认 True，必须有 classifier checkpoint）

**真机指标：**
- Success rate
- Episode return
- Success time
- Episode length
- Average/max action magnitude
- Gripper open/close count
- Safety/truncated/done reason

**输出：**
- `bc_eval/real/`: rollout 视频和 action CSV（如果启用）
- 终端打印汇总指标

**注意事项：**
- 必须先训练 classifier，否则无法自动统计 success
- 必须保持 OpenArm hardware server 运行在 `http://127.0.0.1:5000/`
- 不会写入 `checkpoints_rlpd/buffer` 或 `checkpoints_rlpd/demo_buffer`
- 不启动 TrainerServer/TrainerClient

### 6. 启动 RLPD 训练（自动 BC Warmup）

```bash
# Learner
python train_rlpd.py \
    --exp_name openarm_pickplace \
    --learner

# Actor（另一个终端）
python train_rlpd.py \
    --exp_name openarm_pickplace \
    --actor
```

Actor 会连接 OpenArm hardware server 与真实环境交互；启动 actor 前请确认 server 已运行。

**BC Warmup 自动加载逻辑：**

```
create SACAgentHybridDualArm
if latest RLPD checkpoint exists:
    restore RLPD checkpoint  # 优先级最高
else if BC checkpoint exists:
    restore BC checkpoint    # 冷启动时自动使用
else:
    start from default init  # 随机初始化
```

**关键参数：**
- `--exp_name`: 任务名称
- `--bc_checkpoint_path`: BC checkpoint 目录（默认自动）
- `--bc_checkpoint_step`: BC checkpoint 步数（0 表示 latest）
- 其他 RLPD 参数保持不变

**数据加载：**
- `checkpoints_rlpd/buffer/*.pkl` → replay buffer
- `checkpoints_rlpd/demo_buffer/*.pkl` → online intervention demo buffer
- `demo/collected/success/*.pkl` → offline success demos（每次 learner 启动加载）

**验证 BC 加载：**
启动 learner 时应看到类似日志：
```
[INFO] No RLPD checkpoint found
[INFO] Loading BC warmup from checkpoints_bc/checkpoint_20000
[INFO] BC checkpoint restored successfully
```

## Smoke Test

快速验证所有脚本和路径配置：

```bash
cd rl-serl/examples

# 1. 语法检查
python -m py_compile train_bc.py eval_bc_offline.py eval_bc_real.py train_rlpd.py

# 2. BC 训练（2 步）
python train_bc.py --exp_name openarm_pickplace --train_steps 2 --debug

# 3. 离线评估（1 条轨迹）
python eval_bc_offline.py --exp_name openarm_pickplace --max_trajs 1

# 4. RLPD learner mock 模式（2 步）
python train_rlpd.py --exp_name openarm_pickplace --learner --mock --max_steps 2
```

**注意：**
- Mock 图像是黑图，只用于验证 shape、checkpoint restore、JAX compile 和路径
- 不要把 mock rollout 结果当正式评测

## 完整真机验证流程

```bash
# Terminal 0: OpenArm hardware server
conda activate zy
cd /home/sj/Desktop/zy/rl-serl/rl_robot_infra
python -m robot_servers.openarm_server

# Terminal 1: BC/RLPD workflow
conda activate zy
cd /home/sj/Desktop/zy/rl-serl/examples

# 1. 录 10-20 条 success demo + 5-10 条 failure demo
python record_demos.py --exp_name openarm_pickplace

# 2. 训练并评估 classifier
python train_reward_classifier.py --exp_name openarm_pickplace --num_epochs 100
python eval_reward_classifier.py --exp_name openarm_pickplace

# 3. 训练 BC checkpoint（20000 步）
python train_bc.py --exp_name openarm_pickplace --train_steps 20000

# 4. 离线评估 BC：action error、gripper accuracy、高误差帧分布
python eval_bc_offline.py --exp_name openarm_pickplace

# 5. 真机评估 BC：success rate、夹爪行为、动作抖动、safety
python eval_bc_real.py --exp_name openarm_pickplace --eval_n_trajs 10 --argmax

# 6. 启动 RLPD（自动 BC warmup）
# Terminal 1: Learner
python train_rlpd.py --exp_name openarm_pickplace --learner

# Terminal 2: Actor
python train_rlpd.py --exp_name openarm_pickplace --actor
```

## 参数配置参考

### 任务配置（examples/experiments/tasks.py）

每个任务需要在 `TASK_CONFIGS` 中定义：

```python
"openarm_pickplace": TaskConfig(
    env_name="OpenArmPickPlace-v0",
    batch_size=256,
    max_traj_length=100,
    # ... 其他参数
)
```

### BC 训练超参数

推荐起点（根据任务调整）：

```python
train_steps = 20000              # 训练步数
batch_size = 256                 # batch size（使用任务配置）
checkpoint_period = 5000         # checkpoint 保存间隔
learning_rate = 3e-4            # Adam 学习率（agent 内部）
```

### RLPD 训练超参数（含 BC warmup）

```python
# BC warmup 相关
bc_checkpoint_path = None        # 默认自动检测
bc_checkpoint_step = 0          # 0 表示 latest

# RLPD 训练
max_steps = 1000000             # 总训练步数
batch_size = 256                # 同 BC
utd_ratio = 4                   # update-to-data ratio
demo_ratio = 0.5                # demo/replay 混合比例
```

## 常见问题

### Q: BC checkpoint 和 RLPD checkpoint 兼容吗？

A: 是的。`train_bc.py` 使用和 RLPD 完全相同的 `SACAgentHybridDualArm`，保存的 checkpoint 包含 actor、critic、grasp_critic、temperature，可以直接被 RLPD 加载。

### Q: 为什么不用 BCAgent？

A: `BCAgent` 只包含 actor，且输出 14 维动作；而 hybrid dual SAC 输出 12 维连续动作 + grasp_critic 9 类 joint action。直接用 RLPD 同款 agent 做 BC 避免参数迁移问题。

### Q: BC 训练需要多少 demo？

A: 推荐 10-20 条 success demo 作为起点。Demo 过少会导致 BC overfitting，过多会增加录制成本。可以先用 10 条训练，通过离线/真机评估判断是否需要增加。

### Q: 离线评估和真机评估的区别？

A:
- **离线评估**：逐帧预测动作并与 demo 对比，诊断 action imitation 质量，不连接真机。包含 3D 动作轨迹可视化，直观对比预测与示教动作的空间差异。
- **真机评估**：在真实环境中 rollout BC policy，统计 success rate，验证实际表现。

### Q: BC warmup 会降低 RLPD 探索吗？

A: BC 加快启动但 demo 分布窄时可能降低早期探索。RLPD 仍需要 online intervention 和 demo/replay mixed sampling。通过调整 `demo_ratio` 平衡。

### Q: 如何验证 BC 被正确加载？

A: 启动 learner 时查看日志，应打印 BC checkpoint 路径。也可以比较 BC init vs default init 的早期 reward、intervention 次数和有效 transition 比例。

### Q: Gripper 必须单独训练吗？

A: 是的。Hybrid dual SAC 的夹爪由 `grasp_critic` 决策，只训练 actor 会导致夹爪行为接近随机。BC 训练时必须同时监督 actor（12 维连续动作）和 grasp_critic（9 类 joint action）。

### Q: 评估时必须有 classifier 吗？

A: `eval_bc_real.py` 默认需要 classifier 来自动判断 success。如果没有 classifier checkpoint，脚本会报错并提示先训练 classifier。

## 风险与注意事项

1. **Demo 质量**：BC 完全依赖 demo 质量，噪声 demo 会直接污染 BC policy
2. **Gripper 训练**：必须显式监督 `grasp_critic`，否则夹爪未训练
3. **离线评估局限**：离线评估只能诊断 action imitation 质量，不能替代真机 success rate
4. **Hardware server**：录制 demo、真机评估和 RLPD actor 需要先启动 OpenArm hardware server
5. **Classifier 依赖**：真机评估必须有 classifier，否则无法自动统计 success
6. **路径隔离**：BC checkpoint 和 RLPD checkpoint 严格分开，避免混淆
7. **图像裁剪一致性**：demo pkl、BC 训练、RLPD 训练必须使用相同的 `image_primary` 裁剪（128x128）

## 参考

- 实现计划（已删除）：详细设计文档
- 任务配置：`examples/experiments/tasks.py`
- Artifact 路径：`examples/experiments/artifacts.py`
- Agent 实现：`serl/serl_launcher/.../sac_hybrid_dual.py`
