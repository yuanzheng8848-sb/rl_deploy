# OpenArm gRPC 使用指南

## 问题解决

### 1. 导入错误解决

如果遇到 `ModuleNotFoundError: No module named 'openarm_service_pb2'` 错误，这是因为 Python 无法找到生成的 gRPC 文件。

**解决方案：**
- 确保在正确的目录中运行脚本
- 使用正确的导入路径

### 2. 连接错误解决

如果遇到 `无法连接到 gRPC 服务器` 错误，这是因为服务器没有运行。

**解决方案：**

#### 步骤 1: 启动服务器
```bash
cd /home/sj/moqi_workspace/openarm/openarm_grpc
python start_server.py
```

#### 步骤 2: 运行客户端
```bash
# 在另一个终端中
cd /home/sj/moqi_workspace
python pyroki/left_to_right_control.py
```

## 正确的使用流程

### 1. 启动服务器（必需）
```bash
cd /home/sj/moqi_workspace/openarm/openarm_grpc
python start_server.py
```

### 2. 使用客户端
```python
import sys
sys.path.append("/home/sj/moqi_workspace/openarm")

from openarm_grpc.grpc_client import OpenArmControllerClient

# 创建客户端
controller = OpenArmControllerClient(
    host='localhost',
    port=50051,
    enable_left=True,
    enable_right=True
)

# 使用控制器
left_arm_pos, left_gripper_pos = controller.get_left_position()
```

## 常见问题

### Q: 导入错误怎么办？
A: 确保使用正确的导入路径：
```python
import sys
sys.path.append("/home/sj/moqi_workspace/openarm")
from openarm_grpc.grpc_client import OpenArmControllerClient
```

### Q: 连接失败怎么办？
A: 确保服务器正在运行：
```bash
python start_server.py
```

### Q: 如何检查服务器状态？
A: 查看服务器输出，应该显示：
```
启动 gRPC 服务器，监听端口: 50051
```

## 完整示例

### 服务器端
```bash
# 终端 1
cd /home/sj/moqi_workspace/openarm/openarm_grpc
python start_server.py
```

### 客户端
```bash
# 终端 2
cd /home/sj/moqi_workspace
python pyroki/left_to_right_control.py
```

现在 gRPC 封装已经可以正常工作了！
