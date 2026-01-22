# OpenArm gRPC 远程控制

这个模块提供了 OpenArm 控制器的 gRPC 远程调用封装，允许通过网络远程控制机械臂。

## 功能特性

- **完全兼容的接口**: 客户端接口与原始 `OpenArmController` 完全一致
- **远程控制**: 通过网络远程控制机械臂
- **双臂支持**: 支持同时控制左臂和右臂
- **实时通信**: 基于 gRPC 的高性能通信
- **错误处理**: 完善的错误处理和异常管理

## 文件结构

```
openarm_grpc/
├── openarm_service.proto          # gRPC 服务定义
├── openarm_service_pb2.py         # 生成的 Python 消息类
├── openarm_service_pb2_grpc.py    # 生成的 Python 服务类
├── grpc_server.py                 # gRPC 服务器实现
├── grpc_client.py                 # gRPC 客户端封装
├── start_server.py                # 服务器启动脚本
├── example_client.py              # 客户端使用示例
└── README.md                      # 说明文档
```

## 安装依赖

```bash
pip install grpcio grpcio-tools
```

## 使用方法

### 1. 启动服务器

```bash
# 使用默认端口 50051
python start_server.py

# 指定端口
python start_server.py --port 50052
```

### 2. 使用客户端

#### 基本使用

```python
from grpc_client import OpenArmControllerClient

# 创建客户端
controller = OpenArmControllerClient(
    host='localhost',
    port=50051,
    enable_left=True,
    enable_right=True
)

# 获取位置
left_arm_pos, left_gripper_pos = controller.get_left_position()
right_arm_pos, right_gripper_pos = controller.get_right_position()

# 设置位置
controller.set_left_position(
    target_arm_positions=[-1, 0, 0, 0, 0, 0, -1],
    target_gripper_position=-0.3,
    current_arm_positions=left_arm_pos,
    current_gripper_position=left_gripper_pos
)

# 关闭连接
controller.close()
```

#### 使用上下文管理器

```python
with OpenArmControllerClient(host='localhost', port=50051) as controller:
    # 使用控制器
    left_arm_pos, left_gripper_pos = controller.get_left_position()
    # 自动关闭连接
```

### 3. 运行示例

```bash
python example_client.py
```

## API 接口

### OpenArmControllerClient

#### 构造函数
```python
OpenArmControllerClient(host='localhost', port=50051, enable_left=True, enable_right=True)
```

#### 主要方法

- `get_left_position()`: 获取左臂位置
- `set_left_position(arm_pos, gripper_pos, current_arm_pos, current_gripper_pos)`: 设置左臂位置
- `get_right_position()`: 获取右臂位置  
- `set_right_position(arm_pos, gripper_pos, current_arm_pos, current_gripper_pos)`: 设置右臂位置
- `test_run()`: 运行测试
- `close()`: 关闭连接

## 网络配置

### 服务器端
- 默认监听地址: `0.0.0.0:50051`
- 支持多客户端连接
- 线程池处理并发请求

### 客户端
- 支持指定服务器地址和端口
- 自动重连机制
- 连接池管理

## 错误处理

客户端会自动处理以下错误：
- 网络连接错误
- 服务器未响应
- gRPC 调用失败
- 控制器初始化失败

## 性能优化

- 使用 gRPC 的高性能序列化
- 异步处理多个请求
- 连接复用
- 最小化网络延迟

## 注意事项

1. **网络延迟**: 远程控制会有网络延迟，不适合实时性要求极高的应用
2. **网络稳定性**: 确保网络连接稳定，避免控制中断
3. **安全考虑**: 生产环境中建议使用 TLS 加密
4. **资源管理**: 及时关闭客户端连接，避免资源泄漏

## 故障排除

### 常见问题

1. **连接失败**
   - 检查服务器是否启动
   - 确认端口是否正确
   - 检查防火墙设置

2. **初始化失败**
   - 确认硬件连接正常
   - 检查 CAN 总线状态
   - 查看服务器日志

3. **位置设置失败**
   - 检查目标位置是否合理
   - 确认当前位置数据正确
   - 检查机械臂状态

### 调试模式

启动服务器时添加详细日志：
```bash
python start_server.py --verbose
```

## 扩展功能

- 支持更多控制模式
- 添加安全限制
- 实现位置记录和回放
- 支持多机械臂协调控制
