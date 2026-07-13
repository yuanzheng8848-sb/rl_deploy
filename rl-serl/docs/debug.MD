# OpenArm 新架构修复与验证记录

更新时间：2026-07-13

本文档原先记录了 Franka SERL 迁移到 OpenArm 后的遗留与数据流问题。相关问题现已按 OpenArm 的硬件边界、网络契约和训练数据流重新设计并修复，不再保留旧接口兼容路径。

## 当前架构契约

- `OpenArmEnv` 只负责机器人状态、控制会话和 Gymnasium 契约，不采集或解码网络图像。
- `LocalOpenArmEnv` 独占相机采集；相机硬件来自 `openarm_configs/cameras.yaml`，所有相机通过 `read_rgb()` 返回明确的 RGB 图像。
- `openarm_server.py` 只提供机器人状态、控制和诊断。`/state` 返回 `pose`、`gripper_closed`、`timestamp`，不再返回 Franka 占位字段或图像。
- HTTP 夹爪契约只传 `[left_closed, right_closed]` 布尔状态。原始电机位置、开合目标和滞回阈值只存在于 `openarm_configs/control.yaml`，由 `GripperCalibration` 管理。
- servo 参数和 home 关节位姿只由 `openarm_configs/control.yaml` 定义。环境在 reset 后启动唯一控制会话，在再次 reset 或 close 时停止；`/control/target` 不再隐式启动 servo。
- 基础环境奖励固定为 `0.0` 且不产生 task termination；视觉分类器 wrapper 独占成功奖励和终止判定。
- 遥操作产生 `intervention_transition` 观测/动作片段，奖励与 done 始终由外层环境结果填写，训练和 demo 录制使用同一语义。

## 原问题逐项处理

### 环境与 wrapper

1. 删除未使用的 `TARGET_POSE`、`REWARD_THRESHOLD`。
2. 删除按步位移语义的 `ACTION_SCALE`；统一使用 `ACTION_VELOCITY_SCALE=[0.05 m/s, 0.25 rad/s]`，所有目标积分显式乘 `dt`，动作语义不再依赖控制频率。
3. 删除基础环境的死 `compute_reward` 签名和无效夹爪参数。
4. 基础环境明确返回 `terminated=False`，终止只由奖励分类器负责。
5. 删除环境中的 `q`/`dq` 缓存和无消费的数据解析。
6. 删除 `DualRelativeFrame.step()` 中不可能触发的 `intervene_action` 逆变换。
7. OpenArm wrapper 复用 `rl_launcher.wrappers.chunking.stack_obs`。
8. `OpenArmPolicyObsAdapter` 从实际安装的 wrapper 链动态构建，不再手工复制装配顺序。
9. 删除未使用的 Euler/块变换函数。
10. `openarm_env` 与 `openarm_env.envs` 提供轻量公共导出和版本信息。

### 控制频率与 target 追踪

- 服务端 servo 以 100 Hz 持续追踪最新 Cartesian target。
- 策略和 replay transition 固定为 20 Hz；环境的周期调度包含策略推理耗时，不再在每次 `step()` 内额外等待一个完整旧 5 Hz 周期。
- SpaceMouse 在每个 20 Hz transition 窗口内以 80 Hz 更新 target，四个高频控制 tick 时间加权聚合成一条策略坐标动作。
- `EV_ABS` 输入保持当前轴位置，`EV_REL` 输入在每个 80 Hz tick 消费后清零。
- 回合上限使用秒定义；80 秒对应 1600 个 20 Hz transition。随机探索保持 10 秒，对应 200 个 transition。
- 奖励 wrapper 一次推理同时返回 reward 和 classifier probability，actor 不再为渲染/日志重复加载和执行第二份分类器。

### 服务端与控制

11. `/state` 删除 `vel`、`force`、`torque` 等占位符以及环境不消费的关节/诊断字段；完整硬件信息保留在 `/diagnostics`。
12. 删除无关的 SymPy `print_glsl` 导入。
13. servo debug 初始 backend 直接来自控制配置，与实际 backend 一致。
14. 删除服务端相机线程、base64 图像编码和旧 `/getstate` 路由。
15. 夹爪原始单位收口到 `GripperCalibration`，通过显式标定和滞回阈值判断开合。
16. 策略和遥操作共享同一 servo 会话与同一服务端配置。
17. `/control/target` 不再自动启动 servo，控制生命周期只有环境这一位 owner。
18. home 关节位姿从服务端硬编码迁移到 `openarm_configs/control.yaml`；环境侧仅保留明确命名的虚拟 reset 位姿。

### 训练与介入数据流

19. `sampled_transition` 更名为无伪奖励字段的 `intervention_transition`；actor 与 demo recorder 使用环境返回的 reward/done。
20. actor 通过公开只读属性 `intervention_active` 查询介入状态，不再访问 wrapper 私有字段。
21. 删除被禁用的 `GripperPenaltyWrapper`、replay 字段和 agent 对 `grasp_penalty` 的依赖。
22. 介入 idle 会刷新实时观测；奖励分类器识别 idle 并跳过推理和终止判定。
23. 双 SpaceMouse 的介入切换请求按一轮事件合并，同轮双设备触发只切换一次。

### 相机与图像

24. 非 Local 环境显式拒绝相机配置，不再保留永远无法触发的服务端图像解码路径。
25. `LocalOpenArmEnv` 是 `latest_images` 的唯一写入源，不再存在服务端/本地双源竞态。
26. 批量裁剪从实际输出推导形状，支持非方形 `MODEL_IMAGE_SIZE`。
27. USB、RealSense、Mock 相机统一实现 `read_rgb()`，不再根据返回值类型猜测颜色空间。

## 验证

在 WSL Conda `rl` 环境执行：

```bash
/home/sb/miniconda3/bin/conda run -n rl python -m pytest -q /mnt/d/rl_depoly/rl-serl/tests
```

测试覆盖虚拟环境、真实环境 HTTP 契约、单一 servo 生命周期、夹爪标定滞回、显式相机配置、RGB 接口、动态 wrapper 链、非方形批量裁剪和 intervention idle 奖励行为。
