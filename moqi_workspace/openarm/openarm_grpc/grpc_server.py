import grpc
from concurrent import futures
import threading
import time
import sys
import os

# 添加父目录到路径，以便导入 openarm_controller
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加当前目录到路径，以便导入生成的 gRPC 文件
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from openarm_controller import OpenArmController
import openarm_service_pb2
import openarm_service_pb2_grpc


class OpenArmServicer(openarm_service_pb2_grpc.OpenArmServiceServicer):
    def __init__(self):
        self.controller = None
        self.initialized = False
        self.lock = threading.Lock()

    def Initialize(self, request, context):
        """初始化控制器"""
        try:
            with self.lock:
                if self.initialized:
                    return openarm_service_pb2.InitializeResponse(
                        success=True,
                        message="控制器已经初始化"
                    )
                
                self.controller = OpenArmController(
                    enable_left=request.enable_left,
                    enable_right=request.enable_right
                )
                self.initialized = True
                
                return openarm_service_pb2.InitializeResponse(
                    success=True,
                    message="控制器初始化成功"
                )
        except Exception as e:
            return openarm_service_pb2.InitializeResponse(
                success=False,
                message=f"控制器初始化失败: {str(e)}"
            )

    def GetLeftPosition(self, request, context):
        """获取左臂位置"""
        try:
            if not self.initialized or not self.controller:
                return openarm_service_pb2.GetPositionResponse(
                    success=False,
                    message="控制器未初始化"
                )
            
            if not self.controller.enable_left:
                return openarm_service_pb2.GetPositionResponse(
                    success=False,
                    message="左臂未启用"
                )
            
            arm_positions, gripper_positions = self.controller.get_left_position()
            
            return openarm_service_pb2.GetPositionResponse(
                success=True,
                message="获取左臂位置成功",
                arm_positions=arm_positions,
                gripper_positions=gripper_positions
            )
        except Exception as e:
            return openarm_service_pb2.GetPositionResponse(
                success=False,
                message=f"获取左臂位置失败: {str(e)}"
            )

    def SetLeftPosition(self, request, context):
        """设置左臂位置"""
        try:
            if not self.initialized or not self.controller:
                return openarm_service_pb2.SetPositionResponse(
                    success=False,
                    message="控制器未初始化"
                )
            
            if not self.controller.enable_left:
                return openarm_service_pb2.SetPositionResponse(
                    success=False,
                    message="左臂未启用"
                )
            
            self.controller.set_left_position(
                request.arm_target_positions,
                request.gripper_target_position,
                request.current_arm_positions,
                request.current_gripper_position
            )
            
            return openarm_service_pb2.SetPositionResponse(
                success=True,
                message="设置左臂位置成功"
            )
        except Exception as e:
            return openarm_service_pb2.SetPositionResponse(
                success=False,
                message=f"设置左臂位置失败: {str(e)}"
            )

    def GetRightPosition(self, request, context):
        """获取右臂位置"""
        try:
            if not self.initialized or not self.controller:
                return openarm_service_pb2.GetPositionResponse(
                    success=False,
                    message="控制器未初始化"
                )
            
            if not self.controller.enable_right:
                return openarm_service_pb2.GetPositionResponse(
                    success=False,
                    message="右臂未启用"
                )
            
            arm_positions, gripper_positions = self.controller.get_right_position()
            
            return openarm_service_pb2.GetPositionResponse(
                success=True,
                message="获取右臂位置成功",
                arm_positions=arm_positions,
                gripper_positions=gripper_positions
            )
        except Exception as e:
            return openarm_service_pb2.GetPositionResponse(
                success=False,
                message=f"获取右臂位置失败: {str(e)}"
            )

    def SetRightPosition(self, request, context):
        """设置右臂位置"""
        try:
            if not self.initialized or not self.controller:
                return openarm_service_pb2.SetPositionResponse(
                    success=False,
                    message="控制器未初始化"
                )
            
            if not self.controller.enable_right:
                return openarm_service_pb2.SetPositionResponse(
                    success=False,
                    message="右臂未启用"
                )
            
            self.controller.set_right_position(
                request.arm_target_positions,
                request.gripper_target_position,
                request.current_arm_positions,
                request.current_gripper_position
            )
            
            return openarm_service_pb2.SetPositionResponse(
                success=True,
                message="设置右臂位置成功"
            )
        except Exception as e:
            return openarm_service_pb2.SetPositionResponse(
                success=False,
                message=f"设置右臂位置失败: {str(e)}"
            )

    def TestRun(self, request, context):
        """测试运行"""
        try:
            if not self.initialized or not self.controller:
                return openarm_service_pb2.TestRunResponse(
                    success=False,
                    message="控制器未初始化"
                )
            
            # 在单独线程中运行测试，避免阻塞 gRPC 调用
            def run_test():
                self.controller.test_run()
            
            test_thread = threading.Thread(target=run_test)
            test_thread.daemon = True
            test_thread.start()
            
            return openarm_service_pb2.TestRunResponse(
                success=True,
                message="测试运行已启动"
            )
        except Exception as e:
            return openarm_service_pb2.TestRunResponse(
                success=False,
                message=f"测试运行失败: {str(e)}"
            )


def serve(port=50051):
    """启动 gRPC 服务器"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    openarm_service_pb2_grpc.add_OpenArmServiceServicer_to_server(
        OpenArmServicer(), server
    )
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    print(f"启动 gRPC 服务器，监听端口: {port}")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("正在关闭服务器...")
        server.stop(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenArm gRPC 服务器')
    parser.add_argument('--port', type=int, default=50051, help='服务器端口 (默认: 50051)')
    args = parser.parse_args()
    
    serve(args.port)
