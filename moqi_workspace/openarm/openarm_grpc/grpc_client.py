import grpc
import numpy as np
import time
import sys
import os

# 添加当前目录到路径，以便导入生成的 gRPC 文件
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import openarm_service_pb2
    import openarm_service_pb2_grpc
except ImportError:
    # 如果直接导入失败，尝试从当前目录导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("openarm_service_pb2", 
                                                os.path.join(current_dir, "openarm_service_pb2.py"))
    openarm_service_pb2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(openarm_service_pb2)
    
    spec = importlib.util.spec_from_file_location("openarm_service_pb2_grpc", 
                                                os.path.join(current_dir, "openarm_service_pb2_grpc.py"))
    openarm_service_pb2_grpc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(openarm_service_pb2_grpc)


class OpenArmControllerClient:
    """
    OpenArm 控制器的 gRPC 客户端封装
    提供与原始 OpenArmController 相同的接口
    """
    
    def __init__(self, host='localhost', port=50051, enable_left=True, enable_right=True):
        """
        初始化 gRPC 客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
            enable_left: 是否启用左臂
            enable_right: 是否启用右臂
        """
        self.host = host
        self.port = port
        self.enable_left = enable_left
        self.enable_right = enable_right
        
        # 创建 gRPC 连接
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = openarm_service_pb2_grpc.OpenArmServiceStub(self.channel)
        
        # 初始化控制器
        self._initialize()
    
    def _initialize(self):
        """初始化远程控制器"""
        try:
            request = openarm_service_pb2.InitializeRequest(
                enable_left=self.enable_left,
                enable_right=self.enable_right
            )
            response = self.stub.Initialize(request)
            
            if not response.success:
                raise Exception(f"初始化失败: {response.message}")
            
            print(f"远程控制器初始化成功: {response.message}")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise Exception(f"无法连接到 gRPC 服务器 ({self.host}:{self.port})。请确保服务器正在运行。\n启动服务器: python start_server.py")
            else:
                raise Exception(f"gRPC 调用失败: {e.details()}")
    
    def get_position(self, arm_side):
        """
        获取指定手臂的位置
        
        Args:
            arm_side: "left" 或 "right"
            
        Returns:
            tuple: (arm_positions, gripper_positions)
        """
        try:
            request = openarm_service_pb2.GetPositionRequest(arm_side=arm_side)
            
            if arm_side == "left":
                response = self.stub.GetLeftPosition(request)
            else:
                response = self.stub.GetRightPosition(request)
            
            if not response.success:
                raise Exception(f"获取位置失败: {response.message}")
            
            return list(response.arm_positions), list(response.gripper_positions)
        except grpc.RpcError as e:
            raise Exception(f"gRPC 调用失败: {e.details()}")
    
    def set_position(self, arm_side, arm_target_positions, gripper_target_position,
                    current_arm_positions, current_gripper_position):
        """
        设置指定手臂的位置
        
        Args:
            arm_side: "left" 或 "right"
            arm_target_positions: 目标手臂位置列表
            gripper_target_position: 目标夹爪位置
            current_arm_positions: 当前手臂位置列表
            current_gripper_position: 当前夹爪位置
        """
        try:
            request = openarm_service_pb2.SetPositionRequest(
                arm_target_positions=arm_target_positions,
                gripper_target_position=gripper_target_position,
                current_arm_positions=current_arm_positions,
                current_gripper_position=current_gripper_position,
                arm_side=arm_side
            )
            
            if arm_side == "left":
                response = self.stub.SetLeftPosition(request)
            else:
                response = self.stub.SetRightPosition(request)
            
            if not response.success:
                raise Exception(f"设置位置失败: {response.message}")
        except grpc.RpcError as e:
            raise Exception(f"gRPC 调用失败: {e.details()}")
    
    # 左臂相关方法
    def get_left_position(self):
        """获取左臂位置"""
        return self.get_position("left")
    
    def set_left_position(self, left_arm_position, left_gripper_position,
                         current_arm_position, current_gripper_position):
        """设置左臂位置"""
        self.set_position("left", left_arm_position, left_gripper_position,
                         current_arm_position, current_gripper_position)
    
    # 右臂相关方法
    def get_right_position(self):
        """获取右臂位置"""
        return self.get_position("right")
    
    def set_right_position(self, right_arm_position, right_gripper_position,
                          current_arm_position, current_gripper_position):
        """设置右臂位置"""
        self.set_position("right", right_arm_position, right_gripper_position,
                         current_arm_position, current_gripper_position)
    
    def test_run(self):
        """测试运行"""
        try:
            request = openarm_service_pb2.TestRunRequest()
            response = self.stub.TestRun(request)
            
            if not response.success:
                raise Exception(f"测试运行失败: {response.message}")
            
            print(f"测试运行: {response.message}")
        except grpc.RpcError as e:
            raise Exception(f"gRPC 调用失败: {e.details()}")
    
    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 为了保持与原始 OpenArmController 的完全兼容性
# 创建一个别名类
OpenArmController = OpenArmControllerClient


if __name__ == "__main__":
    # 测试客户端
    try:
        # 创建客户端
        controller = OpenArmControllerClient(
            host='localhost',
            port=50051,
            enable_left=True,
            enable_right=True
        )
        
        print("客户端连接成功")
        
        # 测试获取位置
        if controller.enable_left:
            left_arm_pos, left_gripper_pos = controller.get_left_position()
            print(f"左臂位置: {left_arm_pos}")
            print(f"左夹爪位置: {left_gripper_pos}")
        
        if controller.enable_right:
            right_arm_pos, right_gripper_pos = controller.get_right_position()
            print(f"右臂位置: {right_arm_pos}")
            print(f"右夹爪位置: {right_gripper_pos}")
        
        # 测试设置位置（示例）
        # controller.set_left_position([0, 0, 0, 0, 0, 0, 0], 0.0, left_arm_pos, left_gripper_pos)
        
    except Exception as e:
        print(f"客户端测试失败: {e}")
    finally:
        if 'controller' in locals():
            controller.close()
