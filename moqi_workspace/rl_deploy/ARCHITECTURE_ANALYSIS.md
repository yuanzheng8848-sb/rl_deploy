# rl_deploy 与 hil-serl 架构对比分析报告

> 对比对象：
> - `moqi_workspace/rl_deploy/`（OpenArm 双臂部署代码，自研）
> - `hil-serl/`（HIL-SERL 官方框架）
>
> 结论先行：两者算法内核相同（都用 HIL-SERL 的 `serl_launcher` 做 SAC/RLPD + 人类干预），但**工程架构差异巨大**。`hil-serl` 是分层、可复用的库式架构；`rl_deploy` 是把所有东西塞进一个 2495 行 `train.py` 的单体脚本。`hil-serl` 的架构更合理，`rl_deploy` 可以、也应该被改造成同样的分层结构。

---

## 1. 整体架构对比

### hil-serl（分层库式架构）

```
hil-serl/
├── serl_launcher/              # 算法库（pip install -e .）
│   └── serl_launcher/
│       ├── agents/continuous/  # SAC / RLPD / BC agent
│       ├── networks/           # encoder、reward classifier
│       ├── data/               # replay buffer / data store
│       ├── wrappers/           # 与机器人无关的通用 wrapper（chunking、obs、video）
│       ├── vision/             # 视觉模型
│       └── utils/launcher.py   # make_*_agent / make_trainer_config 工厂
├── serl_robot_infra/           # 机器人硬件层（pip install -e .）
│   ├── robot_servers/          # Flask server（ROS 通信）
│   └── franka_env/
│       ├── envs/franka_env.py  # gym.Env 硬件接口
│       ├── envs/wrappers.py    # 与机器人相关的 wrapper（干预、夹爪、奖励分类器）
│       └── envs/relative_env.py
└── examples/
    ├── train_rlpd.py           # 通用训练入口（actor/learner，~400 行）
    ├── train_bc.py
    ├── record_demos.py
    ├── train_reward_classifier.py
    └── experiments/
        └── <task_name>/
            ├── config.py       # 任务配置 + get_environment() 组装 wrapper 栈
            └── wrapper.py      # 任务特有的 env 子类
```

**三层清晰分离：**

| 层 | 职责 | 是否随任务变化 |
| --- | --- | --- |
| `serl_launcher` | 算法、网络、buffer、通用 wrapper | 否（任务无关） |
| `serl_robot_infra` | 硬件 server + gym env + 机器人 wrapper | 否（机器人无关，跨任务复用） |
| `examples/experiments/<task>` | 任务配置、wrapper 组装、奖励定义 | 是（每个任务一个目录） |

训练入口 `train_rlpd.py` 通过 `CONFIG_MAPPING[exp_name]` 拿到任务配置，调用 `config.get_environment()` 组装环境，自身不含任何任务/硬件逻辑。

### rl_deploy（单体脚本架构）

```
rl_deploy/
├── train.py                 # 2495 行：actor + learner + 6 个 wrapper + 裁剪/掩码/工具函数全塞一起
├── openarm_env.py           # 512 行：gym.Env（混了相机线程、IK、显示）
├── openarm_server.py        # Flask server（硬件 + IK + Viser）
├── mock_hardware.py
├── train_bc_standalone.py   # BC 训练（与 train.py 大量重复）
├── eval_bc_standalone.py
├── classifier/              # 奖励分类器训练/评估
├── demo/record_demo.py      # 演示采集
├── test/                    # 硬件测试脚本
└── run_*.sh                 # 启动脚本
```

**没有分层。** `train.py` 一个文件里同时包含：

- actor 循环、learner 循环（算法层）
- `ChunkingWrapper`、`DualRelativeFrame`、`Quat2EulerWrapper`、`SERLObsWrapper`、`NetworkPrimaryImageCropWrapper`、`ArmFocusWrapper`、`KeyboardRewardWrapper`、`DualSpacemouseIntervention`（wrapper 层，本应在库里）
- 图像裁剪、arm-focus 掩码、夹爪逻辑等几十个工具函数（util 层）
- 任务配置 `TrainOpenArmConfig`（配置层）
- 相机硬件初始化 `LocalOpenArmEnv`（硬件层）

算法、wrapper、配置、硬件四层全部耦合在一两个文件里。

---

## 2. 关键差异逐项对比

| 维度 | hil-serl | rl_deploy |
| --- | --- | --- |
| **代码组织** | 三层分离的 package，`pip install -e .` | 平铺脚本，靠 `sys.path.append` 拼路径 |
| **训练入口** | `train_rlpd.py` ~400 行，纯算法循环 | `train.py` 2495 行，算法 + wrapper + util 混杂 |
| **wrapper 归属** | 通用 wrapper 在 `serl_launcher/wrappers`，机器人 wrapper 在 `franka_env/envs/wrappers.py` | 全部内联在 `train.py` |
| **任务配置** | 每任务一个 `experiments/<task>/config.py`，`get_environment()` 统一组装 | `TrainOpenArmConfig` 硬编码在 `train.py`，wrapper 栈散落在 main 里 |
| **多任务扩展** | 新增任务 = 新建一个 config 目录，注册到 `CONFIG_MAPPING` | 需要复制改 `train.py` 或加分支 |
| **BC / RLPD / 分类器** | 各自独立入口，共享同一套 env/agent 工厂 | `train.py` 与 `train_bc_standalone.py` 大量重复代码 |
| **硬件抽象** | env 只管 gym 接口，硬件细节在 Flask server | `openarm_env.py` 里混了相机线程、IK、cv2 显示 |
| **可测试性** | wrapper / agent 可单独 import 测试 | 想测一个 wrapper 必须 import 整个 train.py（连带 JAX/CUDA 初始化） |
| **依赖管理** | setup.py 声明依赖 | 顶部手写 CUDA 库路径注入 + JAX 兼容性 monkey-patch |

### 一个直观对比：组装环境

**hil-serl**（`ram_insertion/config.py`，声明式、一目了然）：
```python
def get_environment(self, fake_env=False, save_video=False, classifier=False):
    env = RAMEnv(fake_env=fake_env, save_video=save_video, config=EnvConfig())
    env = GripperCloseEnv(env)
    if not fake_env:
        env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    if classifier:
        env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
    return env
```

**rl_deploy**：相同逻辑分散在 `train.py` 的 main 函数中，wrapper 类定义又在同一文件上方几百行处，配置散落各处，难以一眼看全。

---

## 3. 哪个更合理？

**hil-serl 的架构明显更合理**，理由：

1. **关注点分离**：算法、硬件、任务三层解耦。改任务不碰算法，换机器人不碰任务。
2. **可复用**：`serl_launcher` 作为库被所有入口（rlpd/bc/dagger/classifier）共享，没有重复。
3. **可扩展**：新任务只需加一个 `experiments/<task>/` 目录，符合开闭原则。
4. **可测试**：每个 wrapper/agent 是独立模块，能单测，不必启动整个训练栈。
5. **可维护**：单文件 ~400 行 vs 2495 行，定位问题成本天差地别。
6. **声明式组装**：`get_environment()` 让 wrapper 栈一目了然。

rl_deploy 当前结构的代价：`train.py` 和 `train_bc_standalone.py` 重复、wrapper 无法复用、任何修改都要在巨型文件里大海捞针、新增任务只能复制粘贴。

> 注意：rl_deploy 的**算法是对的**——它本来就 import 了 `serl_launcher` 和 `agentlace`。问题纯粹在工程组织：它把本该留在库里的 wrapper/util/config 全下沉到了一个部署脚本里。

---

## 4. 能否改造？—— 能，且改造路径清晰

rl_deploy 已经依赖 `serl_launcher`，所以改造**不是重写算法，而是把 `train.py` 拆解归位**。目标结构：

```
rl_deploy/
├── openarm_env/                    # 新 package（对标 franka_env）
│   ├── envs/
│   │   ├── openarm_env.py          # 纯 gym.Env（从现 openarm_env.py 抽出，去掉相机线程/显示）
│   │   ├── wrappers.py             # DualRelativeFrame / Quat2Euler / DualSpacemouse
│   │   │                           #   / KeyboardReward / ArmFocus / NetworkPrimaryImageCrop
│   │   └── relative_env.py
│   └── camera/                     # LocalOpenArmEnv 的相机线程逻辑独立成模块
├── experiments/
│   ├── config.py                   # DefaultTrainingConfig（对标 hil-serl）
│   ├── mappings.py                 # CONFIG_MAPPING
│   └── openarm_pickplace/          # 第一个任务
│       ├── config.py               # TrainOpenArmConfig + get_environment()
│       └── wrapper.py
├── train_rlpd.py                   # 瘦身后的训练入口（只剩 actor/learner，对标 hil-serl）
├── train_bc.py                     # 复用同一套 env 工厂，消除与 train.py 的重复
├── record_demos.py                 # 从 demo/record_demo.py 迁移
├── train_reward_classifier.py      # 从 classifier/ 迁移
└── openarm_server.py               # 保持（已是 Flask server，对标 franka_server.py）
```

### 改造步骤（建议顺序，每步可独立验证）

1. **抽 wrapper**：把 `train.py` 里 6 个 wrapper 类移到 `openarm_env/envs/wrappers.py`，`train.py` 改为 import。零行为变更，先跑通。
2. **抽 util**：图像裁剪、arm-focus、夹爪等工具函数移到 `openarm_env/utils/`。
3. **抽 config**：建 `experiments/config.py`（`DefaultTrainingConfig`，含 `get_environment()` 抽象方法）和 `experiments/openarm_pickplace/config.py`，把 main 里的 wrapper 组装逻辑搬进 `get_environment()`。
4. **拆 env**：把 `openarm_env.py` 的相机线程/cv2 显示/IK 从 gym.Env 里剥离（对标 franka_env 只保留 gym 接口）。
5. **瘦身入口**：`train.py` → `train_rlpd.py`，只保留 actor/learner 循环 + `CONFIG_MAPPING[exp_name]` 加载，目标 < 500 行。
6. **去重**：`train_bc_standalone.py` 改为复用统一的 env 工厂和 agent 工厂。

### 改造的收益与风险

- **收益**：与上游 hil-serl 结构对齐，未来同步上游更新更容易；新增 OpenArm 任务从“复制 2495 行”变成“加一个 config 目录”；BC/RLPD/classifier 不再重复。
- **风险**：rl_deploy 里有不少 OpenArm 特有逻辑（双臂 arm-focus、键盘奖励、本地相机线程、单臂聚焦掩码）是 hil-serl 单臂 Franka 没有的，迁移时要保证这些行为不丢。**建议每步用 `fake_env`（mock 硬件）回归测试**，确保拆分前后训练行为一致。
- **工作量**：以重构为主，非重写，预计可在不改变任何算法行为的前提下完成。

---

## 5. 小结

| | hil-serl | rl_deploy（现状） | rl_deploy（改造后） |
| --- | --- | --- | --- |
| 算法内核 | serl_launcher | serl_launcher（同） | serl_launcher（不变） |
| 分层 | 三层清晰 | 单体耦合 | 三层清晰 |
| 训练入口行数 | ~400 | 2495 | < 500 |
| 新增任务成本 | 加一个 config 目录 | 复制改巨型脚本 | 加一个 config 目录 |
| 代码重复 | 无 | train.py / train_bc 重复 | 消除 |

**建议**：按上面 6 步增量改造，把 rl_deploy 对齐到 hil-serl 的 `lib / robot_infra / experiments` 三层结构。算法不动，只搬家——风险低、收益高。
