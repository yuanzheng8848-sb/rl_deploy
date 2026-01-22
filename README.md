# OpenArm RL Deployment

本项目实现了基于 SERL (Software for Evolving RL) 框架的 OpenArm 机械臂强化学习部署系统。系统采用分布式架构，分为硬件服务器、训练端 (Learner) 和 执行端 (Actor)。

## 1. 项目结构与代码位置

**核心说明**：本项目所有自定义和新增代码均位于 `moqi_workspace/rl_deploy` 目录下。
-   `moqi_workspace/` 和 `serl/` 目录下的其他代码保持原状，仅修复部分兼容性bug。
-   所有操作（启动 Server、训练、测试）均在 `moqi_workspace/rl_deploy` 目录下进行。

## 2. 系统架构与部署拓扑

系统支持分布式部署，组件分布如下：

| 组件 | 运行位置 | 说明 |
| :--- | :--- | :--- |
| **OpenArm Server** | **机械臂主机** | 直接连接机械臂硬件 (USB/CAN)。 |
| **Actor** (`run_actor.sh`) | **机械臂主机** | 需连接 Server 和 Env，与 Server 同主机运行。 |
| **Learner** (`run_learner.sh`) | **任意主机** | 可以运行在不同主机上，actor通过指定ip地址与learner通信。 |

**IP 配置注意**：
-   启动 **Actor** 时，必须指定 **Learner** 的 IP 地址。
-   修改 `run_actor.sh` 中的 `--ip` 参数：
    ```bash
    python train_rl.py --actor --ip=<LEARNER_HOST_IP> ...
    ```

## 3. 关键组件说明

### 奖励分类器 (Classifiers)
本项目使用了三个不同的奖励分类器 checkpoint，分别用于不同的视角或阶段：

1.  **`classifier_ckpt_cam2_last10`** :
    -   **作用**: 基于 **头顶相机** 的图像判断任务是否成功。
    -   **特点**: 使用更多数据进行训练，整体倾向于输出较低奖励。
2.  **`classifier_ckpt_cam1`** :
    -   **作用**: 基于 **右臂相机** 的图像判断任务是否成功。
    -   **特点**: 判断极为准确。
3.  **`classifier_ckpt`** :
    -   **作用**: 基于 **头顶相机** 的图像判断任务是否成功。
    -   **特点**: 使用少量数据进行训练，整体倾向于输出较高奖励。

### 核心脚本
| 文件名 | 说明 |
| :--- | :--- |
| `openarm_server.py` | 硬件控制服务器，处理 IK 和相机数据。 |
| `openarm_env.py` | Gym 环境定义，处理观测空间 (Images + State) 和动作空间。 |
| `train_rl.py` | RL 训练主入口，包含 Actor 和 Learner 的逻辑。 |
| `run_learner.sh` | 启动 Learner 的脚本，配置了右臂控制 (`--arm=right`)。 |
| `run_actor.sh` | 启动 Actor 的脚本，**需配置 Learner IP**。 |
| `rl_success_demos.pkl` | 成功的演示轨迹数据，用于辅助训练。 |

## 4. 快速开始

启动系统需要打开三个终端，分别运行以下命令：

### 第一步：启动硬件服务器 (机械臂主机)
```bash
# 在 moqi_workspace/rl_deploy 目录下
python openarm_server.py
```

### 第二步：启动 Learner (训练主机)
```bash
# 在 moqi_workspace/rl_deploy 目录下
./run_learner.sh
```

### 第三步：启动 Actor (机械臂主机)
```bash
# 在 moqi_workspace/rl_deploy 目录下
# 务必确认 run_actor.sh 中的 IP 已指向 Learner 主机
./run_actor.sh
```

## 4. 逻辑细节

### 数据流
1.  **Camera/Robot** -> `openarm_server.py` (采集图像和状态)
2.  `openarm_server.py` -> (HTTP JSON) -> `openarm_env.py` (获取观测)
3.  `openarm_env.py` -> `train_rl.py` (Actor) (Step 环境)
4.  `train_rl.py` (Actor) -> (Action) -> `openarm_env.py`
5.  `openarm_env.py` -> (HTTP JSON) -> `openarm_server.py` (发送控制指令)
6.  `openarm_server.py` -> **Robot Hardware** (执行运动)

### 奖励机制
-   **Sparse Reward**: 原始环境默认不提供奖励。
-   **Classifier Reward**: 使用预训练的 Reward Classifier (ResNet) 对当前图像进行打分，作为 RL 的奖励信号。
-   **Demo Reward**: 加载演示数据时，最后 10 步被标记为成功 (Reward=0.5)，其余步骤为路径 (Reward=0.0)。

### 坐标系与控制
-   **观测**: TCP Pose (End-Effector) + Gripper Pos + Images.
-   **动作**: Delta XYZ + Delta RPY + Gripper Command.
-   **转换**: `RelativeFrame` Wrapper 将动作转换为相对于当前 TCP 的增量，`openarm_server` 将其转换为绝对位姿并通过 IK 解算为关节角度。

## 5. 训练监控 (WandB)

本项目使用 **Weights & Biases (WandB)** 进行训练过程的可视化监控。

### 配置步骤
在启动 **Learner** 的主机上，需要先登录 WandB：

1.  **注册/获取 API Key**: 访问 [wandb.ai](https://wandb.ai) 注册账号并获取 API Key。
2.  **本地登录**:
    ```bash
    wandb login <YOUR_API_KEY>
    ```
    或者在运行训练脚本时按照提示输入 Key。

### 查看日志
训练开始后，WandB 会自动在云端创建一个 Project（默认为 `serl_dev`），可以实时查看：
-   **Success Rate**: 任务成功率曲线。
-   **Q-Values**: Critic 网络的 Q 值变化。
-   **Loss**: Actor 和 Critic 的训练损失。
-   **Images**: 训练过程中的图像采样（如果开启了 Image Logging）。