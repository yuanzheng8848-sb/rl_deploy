import numpy as np
import pdb

def closest_points_between_segments(p1, q1, p2, q2):
    """
    计算两条线段之间的最近点。
    返回: (point_on_S1, point_on_S2, distance)
    """
    # 转换为numpy数组以便计算
    p1 = np.array(p1, dtype=float)
    q1 = np.array(q1, dtype=float)
    p2 = np.array(p2, dtype=float)
    q2 = np.array(q2, dtype=float)

    d1 = q1 - p1  # 线段1的方向向量
    d2 = q2 - p2  # 线段2的方向向量
    r = p1 - p2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    e = np.dot(d1, r)
    f = np.dot(d2, r)

    denominator = a * c - b * b

    s = 0.0
    t = 0.0

    # 默认先检查线段是否退化或平行
    if denominator < 1e-6:
        # 线段平行或退化
        # 我们处理退化的情况：如果一条线段退化为点
        if a < 1e-6:
            # 第一条线段退化为点
            s = 0.0
            # 在第二条线段上找到离P1最近的点
            t = f / c
            t = np.clip(t, 0.0, 1.0)
        else:
            # 第一条线段不是点，但两条线段平行
            s = np.clip(-e / a, 0.0, 1.0)
            # 用钳制后的s计算t
            t = (b * s + f) / c
            # 如果第二条线段退化为点(c很小)，直接钳制t
            if c < 1e-6:
                t = 0.0
            else:
                t = np.clip(t, 0.0, 1.0)
    else:
        # 线段不平行，计算初始的s和t
        s = (b * f - c * e) / denominator
        t = (a * f - b * e) / denominator

        # 钳制s到[0, 1]
        if s < 0.0:
            s = 0.0
            t = f / c
            t = np.clip(t, 0.0, 1.0)
        elif s > 1.0:
            s = 1.0
            t = (b + f) / c
            t = np.clip(t, 0.0, 1.0)
        else:
            # s在范围内，现在钳制t
            if t < 0.0:
                t = 0.0
                s = -e / a
                s = np.clip(s, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = (b - e) / a
                s = np.clip(s, 0.0, 1.0)

    # 计算最近点
    point_on_S1 = p1 + s * d1
    point_on_S2 = p2 + t * d2

    # 计算距离
    distance = np.linalg.norm(point_on_S1 - point_on_S2)

    return point_on_S1, point_on_S2, distance

def capsules_intersect(capsule1, capsule2):
    """
    判断两个胶囊体是否相交（碰撞）。
    参数:
        capsule1: ( (P1x, P1y, P1z), (Q1x, Q1y, Q1z), r1 )
        capsule2: ( (P2x, P2y, P2z), (Q2x, Q2y, Q2z), r2 )
    返回: bool
    """
    (p1, q1, r1) = capsule1
    (p2, q2, r2) = capsule2

    # 1. 找到两条线段上的最近点
    _, _, distance = closest_points_between_segments(p1, q1, p2, q2)

    # 2. 判断最短距离是否小于半径之和
    return distance <= (r1 + r2)

class OpenArmCollisionChecker:
    # world 坐标系
    def __init__(self, A_left, A_right, viser_server=None):
        self.body_radius = 0.03
        self.body_p1 = np.array([0, 0, 0])
        self.body_p2 = np.array([0, 0, 0.74])
        self.body_collision = (self.body_p1, self.body_p2, self.body_radius)

        self.A_left = A_left
        self.A_right = A_right
        self.AC_radius = 0.03
        self.BC_radius = 0.03

        if viser_server:
            self.viser_server = viser_server
        
    def draw_capsule_in_viser(self, start_point, end_point, radius, name_space, color=(255, 0, 0)):
        # 计算胶囊体的方向和长度
        direction = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(direction)
        
        if length > 0:
            direction = direction / length
            
            # # 创建圆柱体（主体部分）
            # cylinder = self.viser_server.add_cylinder(
            #     position=((np.array(start_point) + np.array(end_point)) / 2).tolist(),
            #     direction=direction.tolist(),
            #     radius=radius,
            #     length=length,
            #     color=color
            # )
            
            # 创建两个半球体（两端）
            sphere1 = self.viser_server.scene.add_icosphere(
                name = name_space + "sphere1",
                position=start_point,
                radius=radius,
                color=color
            )
            
            sphere2 = self.viser_server.scene.add_icosphere(
                name = name_space + "sphere2",
                position=end_point,
                radius=radius,
                color=color
            )
            
            return sphere1, sphere2


    def check_collision(self, B_left, C_left, B_right, C_right):
        """
        检查自碰撞。
        left BC vs Body
        right BC vs Body
        left BC vs right BC

        返回: [left_body_collision, right_body_collision, left_right_collision]
        """

        collision_result = []

        if self.viser_server:
            self.draw_capsule_in_viser(self.body_p1, self.body_p2, self.body_radius, "body", color=(255, 0, 0))
            self.draw_capsule_in_viser(B_left, C_left, self.BC_radius, "left", color=(0, 255, 0))
            self.draw_capsule_in_viser(B_right, C_right, self.BC_radius, "right", color=(0, 0, 255))

        left_BC_collision = (B_left, C_left, self.BC_radius)
        collision_result.append(capsules_intersect(left_BC_collision, self.body_collision))
        
        right_BC_collision = (B_right, C_right, self.BC_radius)
        collision_result.append(capsules_intersect(right_BC_collision, self.body_collision))

        collision_result.append(capsules_intersect(left_BC_collision, right_BC_collision))

        return collision_result  # 无碰撞