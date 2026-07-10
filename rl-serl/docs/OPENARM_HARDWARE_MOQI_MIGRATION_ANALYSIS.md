# rl-serl OpenArm 硬件层去 moqi_workspace 依赖迁移评估

日期：2026-06-23

## 结论先行

`rl-serl` 目前已经迁移了旧 `moqi_workspace/rl_deploy` 里和训练直接相关的 OpenArm 硬件胶水层：Gym env、local camera env、OpenArm wrappers、Flask hardware server、mock hardware 和测试脚本。它没有再依赖 `moqi_workspace/rl_deploy`。

但它仍然依赖 `moqi_workspace` 里的三类底层硬件能力：

1. `moqi_workspace/openarm`：真实 OpenArm CAN 控制器和 URDF 描述。
2. `moqi_workspace/pyroki`：相机类、IK solver、Viser、workspace constraint。
3. `moqi_workspace/IK`：SpaceMouse servo 使用的 analytic IK 和碰撞检查。

如果目标是“硬件控制部分像 `hil-serl/serl_robot_infra` 一样在 `rl-serl/rl_robot_infra` 内部完备”，需要迁移的不是网络，也不是 `hil-serl/serl_launcher`，而是上述底层硬件能力。最小等价迁移约为：

- 需要新增/搬入约 12 个关键 Python 源文件，约 1,900 行。
- 需要搬入 `pyroki/config` 约 9 个 YAML。
- 需要搬入 OpenArm URDF/SRDF 约 17 个描述文件，约 208 KB。
- 需要搬入 `openarm_can` Python binding 源码/构建文件，最小源码约 24 KB；若要离线可重建，还要带 `subprojects` 约 4.3 MB。
- 需要保留第三方 `pyroki` 包来源，当前本仓已有 `moqi_workspace/third_party/pyroki/src`，18 个 Python 文件，约 532 KB。

推荐迁移目标目录：

```text
rl-serl/rl_robot_infra/
  openarm_env/                  # 已存在：env/wrappers/camera/mock
  robot_servers/                # 已存在：openarm_server.py
  openarm_control/              # 新增：OpenArmController + openarm_can binding 接入
  openarm_ik/                   # 新增：robot_ik_solver / analytic_IK / collision / workspace
  openarm_description/          # 新增：URDF/SRDF/xacro
  openarm_configs/              # 新增：robot.yaml / solver.yaml / viser.yaml
  third_party/pyroki/           # 可选新增：若不依赖环境里 pip-installed pyroki
```

## 当前 rl-serl 硬件层现状

`rl-serl/rl_robot_infra` 当前已有 11 个 Python 文件，体量约 384 KB：

| 文件 | 作用 | 是否已经在 rl-serl 内部 |
|---|---|---|
| `openarm_env/envs/openarm_env.py` | HTTP client Gym env，定义 obs/action、reset、step、gripper 语义 | 是 |
| `openarm_env/envs/local_openarm_env.py` | actor 侧本地相机采集线程 | 是 |
| `openarm_env/envs/wrappers.py` | OpenArm relative frame、quat/euler、crop、gripper penalty、dual SpaceMouse intervention | 是 |
| `openarm_env/camera/local_camera.py` | 三相机硬件常量与构造函数 | 是，但相机类来自 moqi |
| `robot_servers/openarm_server.py` | Flask server、真实控制器、IK、servo、相机、API | 是，但底层控制/IK/相机来自 moqi |
| `openarm_env/mock_hardware.py` | Mock controller/camera | 是 |

当前路径注入位置：

- `examples/compat.py` 把 `moqi_workspace/pyroki` 和 `moqi_workspace/openarm` 加入 `sys.path`。
- `openarm_env/envs/openarm_env.py` 兜底注入 `moqi_workspace/pyroki`。
- `openarm_env/camera/local_camera.py` 兜底注入 `moqi_workspace/pyroki`。
- `robot_servers/openarm_server.py` 使用 `ROOT_DIR = ZY_ROOT / "moqi_workspace"`，再导入 `openarm`、`pyroki`、`IK`。

因此，“抛弃外部导入 `moqi_workspace`”的主要改动集中在 `compat.py`、`local_camera.py`、`openarm_env.py`、`openarm_server.py` 的 import/路径常量。

## 当前 moqi_workspace 依赖图

```text
rl-serl/examples/*
  -> examples/compat.py
     -> hil-serl/serl_launcher              # 网络/agent/replay，可暂时继续复用
     -> hil-serl/serl_robot_infra           # transformations + reward classifier wrapper，可暂时继续复用
     -> moqi_workspace/pyroki               # 需要硬件迁移
     -> moqi_workspace/openarm              # 需要硬件迁移

rl-serl/rl_robot_infra/openarm_env/camera/local_camera.py
  -> moqi_workspace/pyroki/realsense_camera.py
     -> pyrealsense2, cv2

rl-serl/rl_robot_infra/robot_servers/openarm_server.py
  -> moqi_workspace/openarm/openarm_controller_2.py
     -> openarm_can Python extension
  -> moqi_workspace/pyroki/realsense_camera.py
  -> moqi_workspace/pyroki/robot_ik_solver.py
     -> third_party pyroki package, jax, jaxlie, jaxls, yourdfpy
     -> moqi_workspace/openarm/openarm_description/urdf/robot/openarm_bimanual.urdf
     -> moqi_workspace/openarm/openarm_description/urdf/robot/self_collision/openarm.srdf
  -> moqi_workspace/pyroki/viser_base.py
     -> viser, yourdfpy
  -> moqi_workspace/pyroki/workspace_constraint.py
  -> moqi_workspace/IK/analytic_IK.py
  -> moqi_workspace/IK/collision_check.py
```

## 必须迁移清单

### A. 已迁入 rl-serl 的硬件胶水层

这些文件已经在 `rl-serl`，不需要再从 `moqi_workspace/rl_deploy` 搬：

| 文件 | 行数 | 说明 |
|---|---:|---|
| `rl_robot_infra/openarm_env/envs/openarm_env.py` | 519 | OpenArm Gym env / HTTP control client |
| `rl_robot_infra/openarm_env/envs/local_openarm_env.py` | 73 | 本地相机采集线程 |
| `rl_robot_infra/openarm_env/envs/wrappers.py` | 860 | OpenArm-specific wrappers + SpaceMouse intervention |
| `rl_robot_infra/openarm_env/camera/local_camera.py` | 114 | 相机常量和工厂 |
| `rl_robot_infra/robot_servers/openarm_server.py` | 1028 | Flask hardware server |
| `rl_robot_infra/openarm_env/mock_hardware.py` | 74 | mock controller/camera |

### B. 必须从 moqi_workspace 迁入的底层 Python 文件

这些是保持当前真机功能等价的最小 Python 代码集合：

| 当前路径 | 建议迁移到 | 行数 | 必须性 |
|---|---|---:|---|
| `moqi_workspace/openarm/openarm_controller_2.py` | `rl_robot_infra/openarm_control/openarm_controller_2.py` | 274 | 必须，真实双臂 CAN 控制入口 |
| `moqi_workspace/pyroki/realsense_camera.py` | `rl_robot_infra/openarm_env/camera/realsense_camera.py` 或 `openarm_ik/realsense_camera.py` | 155 | 必须，相机类 |
| `moqi_workspace/pyroki/robot_ik_solver.py` | `rl_robot_infra/openarm_ik/robot_ik_solver.py` | 398 | 必须，Cartesian pose -> joints |
| `moqi_workspace/pyroki/workspace_constraint.py` | `rl_robot_infra/openarm_ik/workspace_constraint.py` | 204 | 必须，analytic servo workspace clamp |
| `moqi_workspace/IK/analytic_IK.py` | `rl_robot_infra/openarm_ik/analytic_IK.py` | 368 | 必须，SpaceMouse analytic servo |
| `moqi_workspace/IK/collision_check.py` | `rl_robot_infra/openarm_ik/collision_check.py` | 175 | 必须，analytic servo collision check |
| `moqi_workspace/pyroki/viser_base.py` | `rl_robot_infra/openarm_ik/viser_base.py` | 326 | 建议迁移；若禁用 server 可视化可选 |

合计：约 1,900 行 Python。

`moqi_workspace/IK/utils.py` 当前没有被 `rl-serl/openarm_server.py` 直接导入；只有在后续发现 `analytic_IK.py` 的运行路径动态依赖它时再迁。

### C. 必须迁入的配置和机器人描述

| 当前路径 | 建议迁移到 | 说明 |
|---|---|---|
| `moqi_workspace/pyroki/config/robot.yaml` | `rl_robot_infra/openarm_configs/robot.yaml` | 必须，IK robot description 配置 |
| `moqi_workspace/pyroki/config/solver.yaml` | `rl_robot_infra/openarm_configs/solver.yaml` | 必须，IK 权重/约束配置 |
| `moqi_workspace/pyroki/config/viser.yaml` | `rl_robot_infra/openarm_configs/viser.yaml` | 若保留 Viser 则必须 |
| `moqi_workspace/pyroki/config/calib/*.yaml` | `rl_robot_infra/openarm_configs/calib/` | 当前 server 不直接读，可作为完整硬件配置迁移 |
| `moqi_workspace/openarm/openarm_description/urdf/robot/openarm_bimanual.urdf` | `rl_robot_infra/openarm_description/urdf/robot/openarm_bimanual.urdf` | 必须，IK 加载模型 |
| `moqi_workspace/openarm/openarm_description/urdf/robot/self_collision/openarm.srdf` | 同相对路径 | 必须，disabled collision pairs |
| `moqi_workspace/openarm/openarm_description/urdf/**/*.xacro` | 同相对路径 | 建议，保持描述包可重生成/可追溯 |

当前 `openarm_description/urdf` 有 20 个文件，其中 URDF/SRDF/xacro 约 17 个，体量约 208 KB。建议整体搬 `urdf/`，不要只搬单个 URDF，否则后续校准/重生成会断。

迁移后需要把 `robot.yaml` 的：

```yaml
description:
  package_path: openarm
  urdf_relative_path: openarm_description/urdf/robot/openarm_bimanual.urdf
```

改成指向 `rl_robot_infra/openarm_description` 的包内路径，或在 server 初始化时动态覆盖 `package_path`。

### D. 必须处理的 openarm_can

`openarm_controller_2.py` 直接 `import openarm_can as oa`。这不是纯 Python 文件，而是 `moqi_workspace/openarm/openarm_can/python` 下的 Python binding。

需要二选一：

1. 把 `openarm_can` 做成独立系统依赖，在部署环境里安装，然后 `rl-serl` 不搬 binding 源码。
2. 把 binding 源码随 `rl-serl/rl_robot_infra/openarm_control/openarm_can/` 搬入，并在安装文档/脚本中构建。

若按“硬件层包内完备”标准，推荐搬入：

```text
moqi_workspace/openarm/openarm_can/python/
  pyproject.toml
  meson.build
  requirements.txt
  build.sh
  src/openarm_can.cpp
  subprojects/*.wrap
  subprojects/packagecache/ 或完整 subprojects/
```

最小源码只有约 24 KB；如果要离线可重建，当前 `subprojects` 约 4.3 MB，也应一起搬。

## 可暂时不迁移的内容

### 网络/训练栈

用户明确允许网络部分继续参考/导入 `hil-serl`。因此下面这些不属于本次硬件迁移必须项：

- `hil-serl/serl_launcher/serl_launcher/agents`
- `hil-serl/serl_launcher/serl_launcher/networks`
- `hil-serl/serl_launcher/serl_launcher/data`
- `hil-serl/serl_launcher/serl_launcher/wrappers`
- `hil-serl/serl_launcher/serl_launcher/vision`

`rl-serl/rl_launcher` 当前本来就是 thin forwarding layer，继续保留即可。

### hil-serl 的 Franka 硬件实现

`hil-serl/serl_robot_infra/franka_env` 是 Franka/Robotiq 体系，不应直接复制成 OpenArm 硬件实现。它的价值是结构参考：

```text
hil-serl/serl_robot_infra/
  franka_env/camera/
  franka_env/envs/
  franka_env/spacemouse/
  franka_env/utils/
  robot_servers/
```

`rl-serl` 已经采用类似结构：

```text
rl-serl/rl_robot_infra/
  openarm_env/camera/
  openarm_env/envs/
  robot_servers/
```

缺的是 OpenArm 底座包：controller、IK、description、config、CAN binding。

### moqi_workspace 里的探索脚本和历史备份

这些不应迁：

- `moqi_workspace/rl_deploy/*`：旧集中式训练/部署脚本，`rl-serl` 已迁移其核心功能。
- `moqi_workspace/backup_rl_deploy/*`：历史备份。
- `moqi_workspace/pyroki/main*.py`、`vr.py`、`plot.py`、`diagnose_failures.py`、`data_recorder.py` 等探索/诊断脚本。
- `moqi_workspace/openarm/openarm_teleop/*`：除非后续要把 OpenArm teleop 工具也纳入 `rl-serl`。
- `moqi_workspace/openarm/openarm_grpc/*`：当前 `rl-serl` 使用 Flask server 和本地 controller，不走 gRPC。

## 需要同步修改的 import/路径

### 1. `examples/compat.py`

删除硬件相关 moqi 路径：

```python
ZY_ROOT / "moqi_workspace" / "pyroki",
ZY_ROOT / "moqi_workspace" / "openarm",
```

替换为 `rl-serl/rl_robot_infra` 内部包路径。`hil-serl/serl_launcher` 和 `hil-serl/serl_robot_infra` 可继续保留。

### 2. `openarm_env/camera/local_camera.py`

当前兜底注入 `moqi_workspace/pyroki`，并导入：

```python
from realsense_camera import OpenCVCamera, RealsenseCamera
```

建议改为包内相对导入：

```python
from openarm_env.camera.realsense_camera import OpenCVCamera, RealsenseCamera
```

或更清晰地放到：

```python
from openarm_ik.realsense_camera import OpenCVCamera, RealsenseCamera
```

### 3. `openarm_env/envs/openarm_env.py`

该文件当前导入 `BaseIKSolver` 只是可选兜底，实际 env 主流程不直接用它。可以：

- 删除这段 `moqi_workspace/pyroki` 注入和 `BaseIKSolver` 导入；或
- 改为从包内 `openarm_ik.robot_ik_solver` 导入。

更推荐删除，因为真正 IK 在 server 端执行。

### 4. `robot_servers/openarm_server.py`

这是主要改动点：

当前：

```python
ROOT_DIR = ZY_ROOT / "moqi_workspace"
sys.path.append(str(ROOT_DIR / "openarm"))
sys.path.append(str(ROOT_DIR / "pyroki"))
from realsense_camera import RealsenseCamera
from openarm_controller_2 import OpenArmController
from robot_ik_solver import BaseIKSolver
from viser_base import ViserBase
from workspace_constraint import create_openarm_constraint
```

建议：

```python
INFRA_ROOT = Path(__file__).resolve().parents[1]
from openarm_control.openarm_controller_2 import OpenArmController
from openarm_env.camera.realsense_camera import OpenCVCamera, RealsenseCamera
from openarm_ik.robot_ik_solver import BaseIKSolver
from openarm_ik.viser_base import ViserBase
from openarm_ik.workspace_constraint import create_openarm_constraint
from openarm_ik import analytic_IK, collision_check
```

并把：

```python
cfg_path = ROOT_DIR / "pyroki" / "config"
r_cfg["description"]["package_path"] = str(ROOT_DIR / "openarm")
```

改为：

```python
cfg_path = INFRA_ROOT / "openarm_configs"
r_cfg["description"]["package_path"] = str(INFRA_ROOT)
```

同时把 `robot.yaml` 的 `urdf_relative_path` 配到 `openarm_description/urdf/robot/openarm_bimanual.urdf`。

## 一个必须修正的隐藏问题

`robot_servers/openarm_server.py::_init_cameras()` 对 head camera 使用：

```python
from connection.cameras import OpenCVCamera
```

但当前仓库中没有在 `moqi_workspace` / `rl-serl` / `hil-serl` 搜到 `connection.cameras` 的定义。`local_camera.py` 已经使用了 `moqi_workspace/pyroki/realsense_camera.py` 中的 `OpenCVCamera`。

迁移时建议直接改成：

```python
from openarm_env.camera.realsense_camera import OpenCVCamera
```

否则即使搬完 `moqi_workspace/pyroki`，server 端 head camera 仍可能因为 `connection.cameras` 缺失而失败。这个改动属于“保持功能等价”的必要修复。

## 两档迁移方案

### 方案 1：最小等价硬件迁移

目标：让 `rl-serl` 不再 import `moqi_workspace`，但继续依赖环境中已安装好的第三方库和 `openarm_can`。

迁移内容：

```text
rl_robot_infra/openarm_env/camera/realsense_camera.py
rl_robot_infra/openarm_ik/robot_ik_solver.py
rl_robot_infra/openarm_ik/workspace_constraint.py
rl_robot_infra/openarm_ik/analytic_IK.py
rl_robot_infra/openarm_ik/collision_check.py
rl_robot_infra/openarm_ik/viser_base.py
rl_robot_infra/openarm_control/openarm_controller_2.py
rl_robot_infra/openarm_configs/{robot.yaml,solver.yaml,viser.yaml}
rl_robot_infra/openarm_description/urdf/
```

外部仍要求：

- `openarm_can` 已可 import。
- `pyroki` 已可 import。
- `pyrealsense2`、`jax`、`jaxlie`、`jaxls`、`yourdfpy`、`viser` 等 Python 依赖可用。

优点：迁移小，代码改动集中。

缺点：不是完全自包含，部署环境仍要提前装 `openarm_can` 和 `pyroki`。

### 方案 2：完整 OpenArm 硬件底座迁移

目标：`rl-serl/rl_robot_infra` 像 `hil-serl/serl_robot_infra` 一样，包含本机器人硬件层所需源码、配置、描述和构建入口。

在方案 1 基础上增加：

```text
rl_robot_infra/openarm_control/openarm_can/
  pyproject.toml
  meson.build
  requirements.txt
  build.sh
  src/openarm_can.cpp
  subprojects/...

rl_robot_infra/third_party/pyroki/src/pyroki/
```

并在 `rl_robot_infra/setup.py` 或项目安装说明中补齐依赖：

```text
pyrealsense2
pyyaml
jax
jaxlie
jaxls
jax_dataclasses
yourdfpy
viser
loguru
evdev
meson/ninja/nanobind build deps for openarm_can
```

优点：最接近“抛弃 moqi_workspace 后仍完备”。

缺点：要维护第三方/原生 binding，安装复杂度上升。

## 迁移后验证清单

建议按从轻到重验证：

1. `python -m compileall rl_robot_infra`
2. `python tests/test_camera.py --fake-env`
3. `python tests/test_camera.py`
4. `cd rl_robot_infra/robot_servers && python openarm_server.py`，确认不再出现 `moqi_workspace` 路径和 `connection.cameras` 导入错误。
5. `curl` 或测试脚本调用 `/getstate`。
6. `python tests/test_3dx_connection.py --dual`
7. `python tests/test_3dx_operation.py --detect-only`
8. `python examples/record_demos.py --exp_name openarm_pickplace`
9. `bash examples/run_actor.sh openarm_pickplace ...`

额外建议加一个静态检查：

```bash
rg -n "moqi_workspace|ROOT_DIR = .*moqi|connection\\.cameras" rl-serl
```

目标是硬件路径内没有 `moqi_workspace` import；文档中提到旧路径可以保留，但代码运行路径不应再依赖。

## 推荐执行顺序

1. 新建 `openarm_ik/`、`openarm_control/`、`openarm_description/`、`openarm_configs/`。
2. 先搬 Python 文件和 YAML/URDF，不动训练入口。
3. 修改 `openarm_server.py` import 和 config path。
4. 修改 `local_camera.py` import，并修复 server head camera 的 `connection.cameras`。
5. 修改 `examples/compat.py`，移除 `moqi_workspace/pyroki` 和 `moqi_workspace/openarm`。
6. 跑静态检查，确保没有新的 `moqi_workspace` 硬件 import。
7. 再决定是否把 `openarm_can` 和 `third_party/pyroki` 纳入仓库内完整维护。

## 最终判断

按当前代码，`rl-serl` 已经完成了 `rl_deploy` 层迁移；剩余迁移量主要是底层 OpenArm SDK/IK/相机/描述文件。为了达到和直接依赖 `moqi_workspace` 一样的效果，最少要迁移 `openarm_controller_2.py`、`realsense_camera.py`、`robot_ik_solver.py`、`workspace_constraint.py`、`analytic_IK.py`、`collision_check.py`、相关 config、OpenArm URDF/SRDF，以及处理 `openarm_can` binding。

如果只迁 Python 胶水而不处理 openarm_can、URDF/config、third-party pyroki 来源，代码表面上能 import，但真机 server 的控制入口、相机和 IK 都无法保证与依赖 moqi_workspace 时等价。当前重构后旧控制入口已移除，统一使用 control API。
