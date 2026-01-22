# OpenArm gRPC 快速开始

## 快速启动

### 1. 启动服务器
```bash
cd /home/sj/moqi_workspace/openarm/openarm_grpc
python start_server.py
```

### 2. 运行客户端示例
```bash
# 在另一个终端中
python example_client.py
```

## 基本使用

### 服务器端
```python
# 启动服务器
python start_server.py --port 50051
```

### 客户端
```python
from grpc_client import OpenArmControllerClient

# 创建客户端（接口与原始控制器完全一致）
controller = OpenArmControllerClient(
    host='localhost',
    port=50051,
    enable_left=True,
    enable_right=True
)

# 使用方式与原始 OpenArmController 完全相同
left_arm_pos, left_gripper_pos = controller.get_left_position()
controller.set_left_position(target_pos, target_gripper, current_pos, current_gripper)

# 关闭连接
controller.close()
```

## 兼容性

客户端接口与原始 `OpenArmController` 完全兼容：

```python
# 原始代码
from openarm_controller import OpenArmController
controller = OpenArmController(enable_left=True, enable_right=True)

# 远程代码（只需修改导入）
from grpc_client import OpenArmControllerClient as OpenArmController
controller = OpenArmController(host='localhost', port=50051, enable_left=True, enable_right=True)

# 其余代码完全相同！
```

## 注意事项

1. 确保服务器在客户端之前启动
2. 网络延迟会影响实时性
3. 生产环境建议使用 TLS 加密
