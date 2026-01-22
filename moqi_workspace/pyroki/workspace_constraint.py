"""
工作空间约束模块

防止VR目标位置超出机械臂物理工作空间，提升IK成功率
"""

import numpy as np


class WorkspaceConstraint:
    """机械臂工作空间约束器"""
    
    def __init__(self, max_reach, safety_margin=0.016):
        """
        初始化工作空间约束器
        
        Args:
            max_reach: 最大工作半径 (m)，通常为 l1 + l2
            safety_margin: 安全余量 (m)，防止边界奇异
        """
        self.max_reach = max_reach
        self.safety_margin = safety_margin
        self.effective_reach = max_reach - safety_margin
        
        # 统计信息
        self.total_calls = 0
        self.constrained_calls = 0
        
        print(f"[WorkspaceConstraint] 初始化:")
        print(f"  最大工作半径: {max_reach:.3f} m")
        print(f"  安全余量:     {safety_margin:.3f} m")
        print(f"  有效半径:     {self.effective_reach:.3f} m")
    
    def constrain_position(self, target_position, origin_position):
        """
        将目标位置约束到工作空间内
        
        Args:
            target_position: 目标位置 (3,) 在世界坐标系
            origin_position: 原点位置 (3,) 通常是肩部位置
            
        Returns:
            constrained_position: 约束后的位置 (3,)
        """
        self.total_calls += 1
        
        # 计算相对位置和距离
        relative_pos = target_position - origin_position
        distance = np.linalg.norm(relative_pos)
        
        # 检查是否超出有效半径
        if distance > self.effective_reach:
            self.constrained_calls += 1
            
            # 缩放到有效半径
            scale = self.effective_reach / distance
            constrained_relative = relative_pos * scale
            constrained_position = origin_position + constrained_relative
            
            return constrained_position
        
        return target_position
    
    def constrain_pose(self, target_pose, origin_position):
        """
        约束目标姿态（保持方向，只调整位置）
        
        Args:
            target_pose: 目标姿态 (7,) [qw, qx, qy, qz, x, y, z]
            origin_position: 原点位置 (3,)
            
        Returns:
            constrained_pose: 约束后的姿态 (7,)
        """
        target_position = target_pose[4:7]
        constrained_position = self.constrain_position(target_position, origin_position)
        
        # 保持姿态，只改变位置
        constrained_pose = target_pose.copy()
        constrained_pose[4:7] = constrained_position
        
        return constrained_pose
    
    def constrain_dual_arm(self, left_pose, right_pose, left_origin, right_origin):
        """
        同时约束双臂
        
        Args:
            left_pose: 左臂目标姿态 (7,)
            right_pose: 右臂目标姿态 (7,)
            left_origin: 左肩位置 (3,)
            right_origin: 右肩位置 (3,)
            
        Returns:
            (left_constrained, right_constrained): 约束后的双臂姿态
        """
        left_constrained = self.constrain_pose(left_pose, left_origin)
        right_constrained = self.constrain_pose(right_pose, right_origin)
        
        return left_constrained, right_constrained
    
    def get_stats(self):
        """
        获取统计信息
        
        Returns:
            dict: 统计数据
        """
        constraint_rate = (self.constrained_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        
        return {
            'total_calls': self.total_calls,
            'constrained_calls': self.constrained_calls,
            'constraint_rate': constraint_rate,
            'effective_reach': self.effective_reach,
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n[WorkspaceConstraint] 统计:")
        print(f"  总调用次数:   {stats['total_calls']}")
        print(f"  约束次数:     {stats['constrained_calls']}")
        print(f"  约束率:       {stats['constraint_rate']:.2f}%")
    
    def reset_stats(self):
        """重置统计"""
        self.total_calls = 0
        self.constrained_calls = 0


# 便捷函数
def create_openarm_constraint(l1=0.22, l2=0.216, safety_margin=0.016):
    """
    创建OpenArm专用的工作空间约束器
    
    Args:
        l1: 大臂长度 (m)
        l2: 小臂长度 (m)
        safety_margin: 安全余量 (m)
        
    Returns:
        WorkspaceConstraint: 约束器实例
    """
    max_reach = l1 + l2
    return WorkspaceConstraint(max_reach, safety_margin)


# 使用示例
if __name__ == "__main__":
    print("=" * 60)
    print("工作空间约束器测试")
    print("=" * 60)
    
    # 创建约束器
    constraint = create_openarm_constraint()
    
    # 模拟肩部位置
    left_shoulder = np.array([0.0, 0.2, 0.5])
    right_shoulder = np.array([0.0, -0.2, 0.5])
    
    # 测试1: 正常范围内的目标
    print("\n测试1: 正常范围内")
    target1 = np.array([0.3, 0.2, 0.6])
    constrained1 = constraint.constrain_position(target1, left_shoulder)
    distance1 = np.linalg.norm(target1 - left_shoulder)
    print(f"  原始目标距离: {distance1:.3f} m")
    print(f"  是否约束:     {'是' if not np.allclose(target1, constrained1) else '否'}")
    
    # 测试2: 超出范围的目标
    print("\n测试2: 超出工作空间")
    target2 = np.array([0.8, 0.2, 0.5])
    constrained2 = constraint.constrain_position(target2, left_shoulder)
    distance2_before = np.linalg.norm(target2 - left_shoulder)
    distance2_after = np.linalg.norm(constrained2 - left_shoulder)
    print(f"  原始目标距离: {distance2_before:.3f} m")
    print(f"  约束后距离:   {distance2_after:.3f} m")
    print(f"  是否约束:     {'是' if not np.allclose(target2, constrained2) else '否'}")
    print(f"  方向保持:     {np.allclose(target2-left_shoulder, constrained2-left_shoulder, rtol=1e-2)}")
    
    # 测试3: 双臂姿态约束
    print("\n测试3: 双臂姿态约束")
    left_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.5])  # 超出范围
    right_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.6])  # 正常范围
    
    left_const, right_const = constraint.constrain_dual_arm(
        left_pose, right_pose, left_shoulder, right_shoulder
    )
    
    left_dist_before = np.linalg.norm(left_pose[4:7] - left_shoulder)
    left_dist_after = np.linalg.norm(left_const[4:7] - left_shoulder)
    right_dist = np.linalg.norm(right_pose[4:7] - right_shoulder)
    
    print(f"  左臂原始距离: {left_dist_before:.3f} m → {left_dist_after:.3f} m")
    print(f"  右臂距离:     {right_dist:.3f} m (未改变)")
    print(f"  姿态保持:     {np.allclose(left_pose[:4], left_const[:4])}")
    
    # 打印统计
    constraint.print_stats()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

