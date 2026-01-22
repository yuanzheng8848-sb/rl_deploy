"""
OpenArm gRPC 远程控制模块

提供 OpenArm 控制器的 gRPC 远程调用封装，允许通过网络远程控制机械臂。

主要组件:
- OpenArmControllerClient: gRPC 客户端封装
- OpenArmController: 与原始控制器兼容的别名
"""

from .grpc_client import OpenArmControllerClient, OpenArmController

__version__ = "1.0.0"
__author__ = "OpenArm Team"

__all__ = [
    "OpenArmControllerClient",
    # "OpenArmController"
]
