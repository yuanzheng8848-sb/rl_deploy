#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
左臂控制右臂控制器
实现左臂控制右臂，控制的关节角第四个和夹爪是相同的，其他的关节角都相反
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# 添加路径到sys.path以便直接import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'openarm'))

# 导入OpenArm控制器
from openarm_controller import OpenArmController


class LeftToRightController:
    """左臂控制右臂的控制器类"""
    
    def __init__(self):
        """初始化控制器"""
        print("初始化左臂控制右臂控制器...")
        
        # 初始化OpenArm控制器，启用左右臂
        self.controller = OpenArmController(enable_left=True, enable_right=True)
        
        print("控制器初始化完成")
    
    def map_left_to_right_joints(self, left_joints, left_gripper):
        """
        将左臂关节角度映射到右臂关节角度
        
        Args:
            left_joints: 左臂关节角度数组 (7个关节)
            left_gripper: 左臂夹爪位置
            
        Returns:
            right_joints: 右臂关节角度数组
            right_gripper: 右臂夹爪位置
        """
        # 确保输入是numpy数组
        left_joints = np.array(left_joints)
        
        # 应用关节映射
        right_joints = np.zeros_like(left_joints)
        
        for i in range(len(left_joints)):
            if i == 3:  # 第4个关节（索引3）保持相同
                right_joints[i] = left_joints[i]
            else:  # 其他关节取相反值
                right_joints[i] = -left_joints[i]
        
        # 夹爪位置保持相同
        right_gripper = left_gripper
        
        return right_joints, right_gripper
    
    def get_current_positions(self):
        """
        获取当前左右臂的位置
        
        Returns:
            left_arm_pos, left_gripper_pos: 左臂关节位置和夹爪位置
            right_arm_pos, right_gripper_pos: 右臂关节位置和夹爪位置
        """
        left_arm_pos, left_gripper_pos = self.controller.get_left_position()
        right_arm_pos, right_gripper_pos = self.controller.get_right_position()
        
        return (left_arm_pos, left_gripper_pos), (right_arm_pos, right_gripper_pos)
    
    def control_loop(self):
        """主控制循环"""
        print("开始左臂控制右臂循环...")
        print("按 Ctrl+C 停止控制")
        
        try:
            while True:
                # 获取当前左臂位置
                (left_arm_pos, left_gripper_pos), (right_arm_pos, right_gripper_pos) = self.get_current_positions()
                
                # 将左臂位置映射到右臂目标位置
                right_target_joints, right_target_gripper = self.map_left_to_right_joints(
                    left_arm_pos, left_gripper_pos[0] if left_gripper_pos else 0.0
                )

                self.controller.set_left_position(
                    left_arm_pos,
                    left_gripper_pos[0] if left_gripper_pos else 0.0,
                    left_arm_pos,
                    left_gripper_pos[0] if left_gripper_pos else 0.0
                )
                
                # 控制右臂到目标位置
                self.controller.set_right_position(
                    right_target_joints.tolist(),
                    right_target_gripper,
                    right_arm_pos,
                    right_gripper_pos[0] if right_gripper_pos else 0.0
                )
                
                # 打印当前状态（可选）
                print(f"左臂关节: {[f'{x:.3f}' for x in left_arm_pos]}")
                print(f"右臂目标: {[f'{x:.3f}' for x in right_target_joints]}")
                print(f"左臂夹爪: {left_gripper_pos[0] if left_gripper_pos else 0.0:.3f}")
                print(f"右臂夹爪: {right_target_gripper:.3f}")
                print("-" * 50)
                
                # 控制频率
                time.sleep(0.01)  # 100Hz
                
        except KeyboardInterrupt:
            print("\n收到停止信号，正在退出...")
        except Exception as e:
            print(f"控制循环出现错误: {e}")
        finally:
            print("左臂控制右臂程序结束")
    
    def test_mapping(self):
        """测试关节映射功能"""
        print("测试关节映射功能...")
        
        # 测试数据
        test_left_joints = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        test_left_gripper = 0.3
        
        right_joints, right_gripper = self.map_left_to_right_joints(
            test_left_joints, test_left_gripper
        )
        
        print(f"左臂关节: {test_left_joints}")
        print(f"右臂关节: {right_joints.tolist()}")
        print(f"左臂夹爪: {test_left_gripper}")
        print(f"右臂夹爪: {right_gripper}")
        
        # 验证映射规则
        print("\n验证映射规则:")
        for i in range(len(test_left_joints)):
            if i == 3:
                print(f"关节{i+1}: 相同 ({test_left_joints[i]:.3f} -> {right_joints[i]:.3f})")
            else:
                print(f"关节{i+1}: 相反 ({test_left_joints[i]:.3f} -> {right_joints[i]:.3f})")
        
        print(f"夹爪: 相同 ({test_left_gripper:.3f} -> {right_gripper:.3f})")


def main():
    """主函数"""
    print("=" * 60)
    print("左臂控制右臂控制器")
    print("=" * 60)
    
    # 创建控制器实例
    controller = LeftToRightController()
    
    # 测试映射功能
    controller.test_mapping()
    
    print("\n" + "=" * 60)
    print("开始实际控制...")
    print("=" * 60)
    
    # 开始控制循环
    controller.control_loop()


if __name__ == "__main__":
    main()
