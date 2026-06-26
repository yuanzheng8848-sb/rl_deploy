# rl_deploy → HIL-SERL 风格架构改造计划

> 目标：把 `moqi_workspace/rl_deploy/` 从单体脚本（`train.py` 2495 行）重构为 hil-serl 的三层分离架构（库 / 硬件 / 任务配置），同时**删除以下四个功能**：
> 1. **Arm-Focus（单臂聚焦）**
> 2. **KeyboardRewardWrapper（键盘奖励标注）**
> 3. **Handoff-Focus（交接聚焦标注与重放）**
> 4. **Episode 统计指标（joint distance / 单调性 / autonomy 等诊断指标）**
>
> 原则：**重构搬家 + 定点删除**，不改动 SAC/RLPD 算法内核（继续复用 `serl_launcher`、`agentlace`）。每一步用 `--mock` fake_env 回归，保证行为可验证。

## 已确认的关键决策

1. **奖励来源**：删除键盘奖励后，奖励改为来自**视觉奖励分类器**（rl_deploy 已有 `classifier/` 训练 + 推理逻辑），对接 hil-serl 的 `MultiCameraBinaryRewardClassifierWrapper`。详见 §3 的对接细节。
2. **SpaceMouse 双臂干预（`DualSpacemouseIntervention`）保留**——hil-serl 本身也有干预机制，actor 里的干预计数/demo buffer 注入逻辑一并保留。
3. **迁移方式**：**新建目标文件夹/文件并迁移，旧文件全部保留**，不在原 `train.py` 上原地删改。改造期间新旧并存，验证通过后由用户决定是否清理旧文件。

---

## 0. 现状回顾（耦合点定位）

`train.py` 当前混杂了四层职责，待删功能的耦合点如下：

| 待删功能 | 代码位置 | 说明 |
| --- | --- | --- |
| Arm-Focus | [train.py:575-733](train.py#L575-L733)（`get_train_arm_mode`/`arm_focus_*`/`mask_action_for_arm_focus`/`black_inactive_camera_for_arm_focus`/`ArmFocusWrapper` 等约 10 个函数 + 1 个 wrapper） | 当前 `arm_focus_enabled()` 恒为 False，是死代码脚手架 |
| KeyboardRewardWrapper | [train.py:762-929](train.py#L762-L929) | 键盘 SPACE/ENTER 打标 + pynput 监听 + render 监视窗 |
| Handoff-Focus | [train.py:1884-1944](train.py#L1884)、actor 循环 [train.py:2153-2155](train.py#L2153)、flags `handoff_keyboard_enabled`/`handoff_demo_repeat` [train.py:1529-1538](train.py#L1529)、`is_handoff_focus_active` [train.py:1607](train.py#L1607)、`<H>` 键监听 [train.py:892-895](train.py#L892) | 与 KeyboardRewardWrapper 强耦合（H 键在它里面） |
| Episode 统计指标 | actor 循环 [train.py:2065-2071](train.py#L2065) + [train.py:2200-2240](train.py#L2200) | `success_history`/`distance_traj`/`min_joint_distance`/`distance_monotonicity`/`autonomy_rate` 等 |

依赖关系要点：
- **Handoff 依赖 KeyboardRewardWrapper**（`<H>` 键在键盘监听里）→ 删 KeyboardReward 时 handoff 触发入口一并消失。
- **删掉 KeyboardRewardWrapper 后，奖励来源必须改为视觉奖励分类器**（对标 hil-serl 的 `MultiCameraBinaryRewardClassifierWrapper`）。rl_deploy 已有 `classifier/` 模块，需接上。这是本次改造**最关键的功能性变更**，需单独确认。
- **Arm-Focus 在 actor 循环也有引用**（[train.py:2116-2117](train.py#L2116) `arm_focus_action`、[train.py:2158](train.py#L2158)），删 wrapper 后这些分支同步删除。

---

## 1. 目标目录结构

> 下列为**新增**文件/目录。旧文件（`train.py`、`openarm_env.py`、`classifier/`、`train_bc_standalone.py`、`eval_bc_standalone.py`、`demo/`、旧 `run_*.sh` 等）**全部原样保留**，改造期间新旧并存。

```
rl_deploy/
├── openarm_env/                      # 【新】硬件层 package（对标 serl_robot_infra/franka_env）
│   ├── __init__.py
│   ├── envs/
│   │   ├── openarm_env.py            # 纯 gym.Env（复制自旧 openarm_env.py，剥离相机线程/cv2 显示）
│   │   ├── wrappers.py               # 保留的 wrapper：DualRelativeFrame / Quat2EulerWrapper
│   │   │                             #   / DualSpacemouseIntervention / GripperPenaltyWrapper
│   │   │                             #   / NetworkPrimaryImageCropWrapper
│   │   └── relative_env.py           # （可选）DualRelativeFrame 独立文件
│   └── camera/
│       └── local_camera.py           # LocalOpenArmEnv 的本地三相机采集线程
├── experiments/                      # 【新】任务配置层（对标 examples/experiments）
│   ├── __init__.py
│   ├── config.py                     # DefaultTrainingConfig（抽象基类 + get_environment 抽象方法）
│   ├── mappings.py                   # CONFIG_MAPPING = {"openarm_pickplace": TrainConfig}
│   └── openarm_pickplace/
│       ├── __init__.py
│       ├── config.py                 # EnvConfig + TrainConfig.get_environment()（组装 wrapper 栈 + 奖励分类器）
│       └── wrapper.py                # 任务特有 env 子类（如需要）
├── train_rlpd.py                     # 【新】训练入口：仅 actor/learner 循环 + CONFIG_MAPPING 加载，目标 < 500 行
├── train_bc.py                       # 【新】复用同一 env/agent 工厂（功能对应旧 train_bc_standalone.py）
├── record_demos.py                   # 【新】演示采集（迁移自 demo/record_demo.py）
├── train_reward_classifier.py        # 【新】迁移自 classifier/train_classifier.py
├── compat.py                         # 【新】CUDA 路径注入 + JAX monkey-patch（抽出自 train.py 顶部）
├── openarm_server.py                 # 保留不动（已是 Flask server，对标 franka_server.py）
└── run_rlpd_*.sh                     # 【新】启动脚本指向 train_rlpd.py（旧 run_*.sh 保留）
```

> `SERLObsWrapper` / `ChunkingWrapper` 不在新结构里重新定义，直接 import `serl_launcher.wrappers`（hil-serl 已提供）。

---

## 2. 分步改造计划

> **迁移模式**：所有改动写入新文件，旧 `train.py` / `classifier/` / `train_bc_standalone.py` 全程保留不动。
> 「删除四功能」= 在迁移到新代码时**不复制**它们，而非编辑旧文件。
> 每一步结束用 `python train_rlpd.py --learner --mock`（或对应等价命令）跑通后再进入下一步；旧入口始终可用作对照。
>
> 顺序逻辑：先搭基础设施（compat → env → wrapper），再建配置层（experiments），最后写新入口时自然只接保留功能。Arm-Focus/Handoff/Episode 指标因「不迁移」而消失，无需单独的删除步骤；KeyboardReward 因奖励改走 classifier 而被替换。

### Step 1 — 抽离运行时兼容补丁到 `compat.py`
- 把 [train.py:26-73](train.py#L26-L73) 与 [eval_classifier.py:32-56](classifier/eval_classifier.py#L32-L56) 重复的 CUDA 库路径注入 + `jax.tree_map`/`tree_leaves`/`ShapedArray.update` patch 统一到新文件 `compat.py`。
- 新入口/脚本第一行 `import compat` 完成副作用。
- **验证**：import 不报错，JAX 能初始化 GPU。

### Step 2 — 抽离 env 与本地相机到 `openarm_env/`
- 现 `openarm_env.py` 的 gym.Env 复制到 `openarm_env/envs/openarm_env.py`，剥离 cv2 显示等非接口逻辑（对标 franka_env 只留 gym 接口）。
- `LocalOpenArmEnv` 的相机线程（[train.py:211-328](train.py#L211)）迁到 `openarm_env/camera/local_camera.py`。
- **不迁移**：与 Arm-Focus 相关的相机涂黑逻辑（`black_inactive_camera_for_arm_focus` 等）。
- **验证**：mock 相机初始化正常，obs 形状与旧实现一致。

### Step 3 — 迁移保留的 wrapper 到 `openarm_env/envs/wrappers.py`
- **迁移（保留）**：`DualRelativeFrame`、`Quat2EulerWrapper`、`DualSpacemouseIntervention`、`GripperPenaltyWrapper`、`NetworkPrimaryImageCropWrapper` 及其依赖的图像裁剪工具（`crop_rgb_image`/`crop_primary_image_for_network` 等）。
- **优先复用库**：`SERLObsWrapper`、`ChunkingWrapper` 改为直接 import `serl_launcher.wrappers`（hil-serl 已有），不再保留 train.py 内的本地副本（[train.py:191-209](train.py#L191)、[train.py:736-759](train.py#L736)）；若签名不兼容则暂留本地版。
- **不迁移（= 删除）**：
  - `ArmFocusWrapper` 及所有 `arm_focus_*` / `mask_action_for_arm_focus` / `black_inactive_camera_for_arm_focus` / `get_train_arm_mode` / `_resolve_gripper_holds` 等（[train.py:575-733](train.py#L575-L733)）。
  - `KeyboardRewardWrapper`（[train.py:762-929](train.py#L762-L929)，含 pynput 监听、render 窗、`<H>` 键）——奖励改由 Step 4 的 classifier 提供。
  - Handoff 相关：`is_handoff_focus_transition` / handoff 加载筛选（[train.py:1884-1944](train.py#L1884)）、`is_handoff_focus_active`（[train.py:1607](train.py#L1607)）。
- **验证**：单独 import 新 wrappers 模块无误；用新 wrapper 栈（暂时无 reward wrapper）`--mock` 组装环境，obs/action space 与旧实现一致。

### Step 4 — 删除 KeyboardRewardWrapper，**首次接入**视觉奖励分类器
> 这是本次改造**唯一的功能性变更**，也是最高风险点。

**重要事实（已核实）**：当前 `train.py` **完全没有调用 classifier**——奖励 100% 来自键盘（`KeyboardRewardWrapper`）。`classifier/train_classifier.py` 和 `classifier/eval_classifier.py` 是**离线独立**训练/评估脚本，从未接入训练环路。所以这不是「切换」奖励来源，而是**第一次把分类器接进 actor**。

**奖励链路对接（核心工作）**：
- 在新的 `experiments/openarm_pickplace/config.py` 的 `get_environment(..., classifier=False)` 里，新增分类器分支（对标 hil-serl `ram_insertion/config.py:115-128`）：
  ```python
  if classifier:
      clf = load_classifier_func(
          key=jax.random.PRNGKey(0),
          sample=env.observation_space.sample(),
          image_keys=self.classifier_keys,             # 见下方 key 适配
          checkpoint_path=os.path.abspath("classifier/classifier_ckpt"),
      )
      def reward_func(obs):
          sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
          return int(sigmoid(clf(obs)) > 0.85)          # 阈值对齐现有 eval（0.5）或调高更稳
      env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
  ```

**三处必须解决的不一致（关键）**：
1. **obs key 名不一致**：训练环境图像 key 是 `image_primary`/`image_left`/`image_right`，而现有 classifier 训练时用的 key 是 `image_0`（单相机，源自 `image_primary`）。`load_classifier_func` 的 `sample` 和推理输入都要按 classifier 训练时的 key 组织——需在 `reward_func` 里做一层 `{"image_0": obs["image_primary"]}` 的重映射，或重新训练 classifier 使其直接吃 `image_primary`。
2. **相机数量不一致**：classifier 是**单相机**（只看 `image_primary`），而环境是三相机。沿用单相机奖励即可（不需要 `image_left/right`），`classifier_keys = ["image_primary"]`。
3. **obs 形状/chunking**：classifier 训练用 `(1,128,128,3)`；环境经 `ChunkingWrapper` 后 obs 带 horizon 维。`MultiCameraBinaryRewardClassifierWrapper` 在 chunking 之后调用，要确认传入 `reward_func` 的 obs 形状与 classifier 期望一致（`fix_image_shape` 的逻辑需移植到 reward_func）。

**删除部分**：
- 删除 `KeyboardRewardWrapper`（[train.py:762-929](train.py#L762-L929)，含 pynput 监听、render 监视窗、`<H>` 键）——不迁移到新代码。
- 新代码不再定义 flags `render`、`handoff_keyboard_enabled`（仅服务该 wrapper）。
- 新 actor 循环不含 `reward_wrapper = find_wrapper(env, KeyboardRewardWrapper)`（[train.py:2051](train.py#L2051)）及其 done/reward 注入；reward 直接由 `MultiCameraBinaryRewardClassifierWrapper` 在 `env.step` 内产生。

**前置条件**：需要一个已训练好的 classifier checkpoint（`classifier/classifier_ckpt/` 已存在，但要确认其 image key 与对接方案一致；若 key 不匹配，需用 `train_classifier.py` 重训或在 reward_func 做重映射）。

**验证**：
- 先用 mock reward_func（恒返回 0 或基于步数）跑通 actor，确认 wrapper 链路无形状错误。
- 再接真实 classifier，用一条已采集轨迹回放，确认末帧 reward=1、首帧 reward=0（对齐 `eval_classifier.py` 的行为）。

### Step 4b — 迁移 classifier 脚本到新结构
- `classifier/train_classifier.py` → `train_reward_classifier.py`（对标 hil-serl `examples/train_reward_classifier.py`）。
- `classifier/eval_classifier.py` 保留为评估工具，路径常量改为相对新结构。
- 把 `eval_classifier.py` 顶部的 CUDA/JAX patch 改为 `import compat`（复用 Step 1 成果），消除重复。
- **验证**：分类器训练脚本能在新路径下加载 `demo/collected/success` 并训练出 checkpoint。

### Step 5 — 删除 Episode 统计指标
- 删除 actor 循环中：`success_history` / `episode_idx` / `last_joint_distance` / `episode_initial_distance` / `min_joint_distance` / `distance_traj`（[train.py:2065-2071](train.py#L2065)）。
- 删除 episode 结束时的指标计算块（[train.py:2200-2240](train.py#L2200)）：`final_joint_distance` / `min_joint_distance` / `distance_reduction_ratio` / `distance_monotonicity` / `mean_step_reward` / `autonomy_rate` / `success_rate_100` / `time_to_success` 等。
- **保留**对标 hil-serl 的最小 episode 统计：`RecordEpisodeStatistics` 自带的 return/length，以及 `intervention_count`/`intervention_steps`（hil-serl actor 本身也记录干预）。
- **验证**：`send-stats` 仍能正常发送精简后的 stats，wandb 不报 key 错误。

### Step 5 — 建立 experiments 配置层
- 新建 `experiments/config.py`（`DefaultTrainingConfig`，含 `image_keys`/`classifier_keys`/`proprio_keys`/`get_environment()` 抽象方法，对标 hil-serl `examples/experiments/config.py`）。
- 新建 `experiments/openarm_pickplace/config.py`：把旧 `create_env` 的 wrapper 组装逻辑（[train.py:1701-1756](train.py#L1701)）搬进 `get_environment()`，**只组装保留的 wrapper + Step 4 的 classifier 奖励分支**（不含 KeyboardReward / ArmFocus）。
- 常量迁入该 config：`TrainOpenArmConfig`（[train.py:135-140](train.py#L135)）、`TRAINING_IMAGE_KEYS`、`PROPRIO_KEYS`、`classifier_keys=["image_primary"]`。
- 新建 `experiments/mappings.py`：`CONFIG_MAPPING = {"openarm_pickplace": TrainConfig}`。
- **验证**：`config.get_environment(fake_env=True, classifier=False)` 能组装出与 Step 3 一致的环境；`--exp_name openarm_pickplace` 能正确路由。

### Step 6 — 新训练入口 `train_rlpd.py`（含丢弃 Episode 指标）
- 新建 `train_rlpd.py`，从旧 `train.py` 迁移 `actor()` / `learner()` / `main()` / flags / `create_agent` / buffer 加载逻辑。
- env 创建改为 `config = CONFIG_MAPPING[FLAGS.exp_name]; env = config.get_environment(...)`。
- **迁移 actor 循环时丢弃以下内容（= 删除 Episode 指标 + 残留耦合）**：
  - 统计变量 `success_history` / `episode_idx` / `last_joint_distance` / `episode_initial_distance` / `min_joint_distance` / `distance_traj`（[train.py:2065-2071](train.py#L2065)）。
  - episode 结束的指标计算块（[train.py:2200-2240](train.py#L2200)）：`final_joint_distance` / `distance_reduction_ratio` / `distance_monotonicity` / `mean_step_reward` / `autonomy_rate` / `success_rate_100` / `time_to_success`。
  - handoff 标记与重复插入（[train.py:2059-2060](train.py#L2059)、[train.py:2153-2155](train.py#L2153)、[train.py:2271-2276](train.py#L2271)）、`arm_focus_action` 分支（[train.py:2116-2117](train.py#L2116)、[train.py:2158](train.py#L2158)）。
  - flags `render` / `handoff_keyboard_enabled` / `handoff_demo_repeat`（[train.py:1529-1538](train.py#L1529)）。
- **保留**对标 hil-serl 的最小统计：`RecordEpisodeStatistics` 的 return/length + `intervention_count`/`intervention_steps`（干预保留，hil-serl actor 本身也记录）。
- 目标行数 < 500。
- **验证**：actor / learner / eval 三种模式 `--mock` 全跑通；`send-stats` 不含已删 key，wandb 不报错；对照 §4 回归基线，obs/action 形状与 buffer 写入一致。

### Step 7 — 去重 BC 与迁移辅助脚本
- 新建 `train_bc.py`：复用 `config.get_environment()` 与统一 agent 工厂，替代 `train_bc_standalone.py`（旧文件保留）。
- 新建 `record_demos.py`（迁移自 `demo/record_demo.py`）；`train_reward_classifier.py` 已在 Step 4b 建立。
- 新建 `run_*.sh` 或更新副本指向 `train_rlpd.py`（旧脚本保留）。
- **验证**：BC 训练/评估、demo 采集脚本各自 `--mock` 或小样本跑通。

---

## 3. 奖励链路对接细节与遗留确认点

**已确认（见顶部「已确认的关键决策」）**：奖励来自视觉分类器；干预保留；新建迁移、旧文件保留。

**开始 Step 4 前仍需逐一核实的技术点**：

1. **classifier checkpoint 的 image key 是否与环境一致？**
   现有 `classifier_ckpt/` 是用 `image_0`（单相机，源自 `image_primary`）训练的。两条路线二选一：
   - 路线 A（省事）：在 `reward_func` 里做 `{"image_0": obs["image_primary"]}` 重映射 + 移植 `fix_image_shape`，复用现有 checkpoint。
   - 路线 B（更干净）：用 `train_reward_classifier.py` 重训，让 classifier 直接吃 `image_primary` key。
   建议先试路线 A，跑通后再考虑是否重训。

2. **chunking 后的 obs 形状**：`MultiCameraBinaryRewardClassifierWrapper` 在 `ChunkingWrapper` 之后调用，传给 `reward_func` 的图像带 horizon 维（如 `(1,128,128,3)` 或 `(1,1,128,128,3)`）。需确认与 classifier 期望输入一致，必要时在 reward_func 内 squeeze。

3. **成功阈值**：现有 `eval_classifier.py` 用 0.5 判定；hil-serl 示例用 0.85 更保守。接入时定一个并记录（建议 0.85 起步，按实际 false-positive 调整）。

4. **奖励频率 / done 语义**：`MultiCameraBinaryRewardClassifierWrapper` 中 `done = done or rew`，即分类器判成功立即结束 episode。需确认这与 OpenArm 任务期望一致（原键盘逻辑是 ENTER 成功 / SPACE 失败都结束）。

---

## 4. 风险与回归策略

- **最大风险**：首次接入 classifier 的奖励链路（Step 4）——obs key / 形状 / 阈值三处不一致是主要坑，必须按 §3 逐项核实。
- **新旧并存**：全程保留旧 `train.py`、`classifier/`、`train_bc_standalone.py` 等不动。新代码写在 `openarm_env/`、`experiments/`、`train_rlpd.py` 等新文件里，可随时对照旧实现，验证通过前不删任何旧文件。
- **回归基线**：动手前先记录一次旧 `train.py --mock` 跑 N 步的 obs/action 形状、buffer 写入条数、wandb stats keys，作为新入口的对照标准。
- **删除即「不迁移」**：Arm-Focus / KeyboardReward / Handoff / Episode 指标这四项，做法是**在迁移时不复制到新代码**，而非修改旧文件——旧 train.py 里它们原样保留。
- **不动算法**：`make_sac_pixel_agent_hybrid_dual_arm`、learner 的 critic/actor 更新逻辑全程不改。

---

## 5. 改造前后对照

| 维度 | 改造前 | 改造后 |
| --- | --- | --- |
| 分层 | 单体 train.py 2495 行 | 库 / 硬件 / experiments 三层 |
| 训练入口行数 | 2495 | < 500（train_rlpd.py） |
| 奖励来源 | 键盘人工打标 | 视觉奖励分类器（对标 hil-serl） |
| Arm-Focus | 死代码脚手架 | 删除 |
| Handoff-Focus | 键盘触发 + demo 重放 | 删除 |
| Episode 指标 | 10+ 项自定义诊断 | 精简为 return/length/intervention |
| 新增任务成本 | 复制改巨型脚本 | 加一个 experiments/<task>/ 目录 |
| 代码重复 | train.py / train_bc 重复 | 共享 env/agent 工厂 |
