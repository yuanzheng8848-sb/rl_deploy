# 左臂控制右臂控制器

## 功能描述

这个Python程序实现了左臂控制右臂的功能，其中：
- 第4个关节（索引3）和夹爪保持与左臂相同
- 其他所有关节角度都与左臂相反

## 文件说明

- `left_to_right_control.py`: 主要的控制程序

## 使用方法

### 1. 直接运行
```bash
cd /home/sj/moqi_workspace/pyroki
python3 left_to_right_control.py
```

### 2. 作为模块导入
```python
from left_to_right_control import LeftToRightController

# 创建控制器
controller = LeftToRightController()

# 测试映射功能
controller.test_mapping()

# 开始控制循环
controller.control_loop()
```

## 关节映射规则

| 关节 | 映射规则 | 说明 |
|------|----------|------|
| 关节1 | 相反 | right_joint[0] = -left_joint[0] |
| 关节2 | 相反 | right_joint[1] = -left_joint[1] |
| 关节3 | 相反 | right_joint[2] = -left_joint[2] |
| 关节4 | 相同 | right_joint[3] = left_joint[3] |
| 关节5 | 相反 | right_joint[4] = -left_joint[4] |
| 关节6 | 相反 | right_joint[5] = -left_joint[5] |
| 关节7 | 相反 | right_joint[6] = -left_joint[6] |
| 夹爪 | 相同 | right_gripper = left_gripper |

## 控制频率

程序以100Hz的频率运行控制循环，确保实时响应。

## 停止程序

按 `Ctrl+C` 可以安全停止程序。

## 依赖项

- numpy
- openarm_controller (来自openarm模块)
- 需要CAN总线连接（can0用于左臂，can1用于右臂）

## 注意事项

1. 确保OpenArm硬件已正确连接
2. 确保CAN总线设备可用
3. 程序会自动初始化左右臂控制器
4. 控制过程中会实时显示关节位置信息
