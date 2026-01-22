"""
IK性能监控模块测试示例

演示如何使用 ik_performance_monitor 模块
"""

import time
import random
from ik_performance_monitor import create_monitor


def simulate_ik_solve(difficulty='easy'):
    """
    模拟IK求解过程
    
    Args:
        difficulty: 'easy', 'medium', 'hard'
    
    Returns:
        tuple: (elapsed_time, solved)
    """
    # 模拟不同难度的求解时间
    if difficulty == 'easy':
        time.sleep(random.uniform(0.001, 0.005))  # 1-5ms
        solved = random.random() > 0.1  # 90%成功率
    elif difficulty == 'medium':
        time.sleep(random.uniform(0.005, 0.015))  # 5-15ms
        solved = random.random() > 0.2  # 80%成功率
    else:  # hard
        time.sleep(random.uniform(0.015, 0.030))  # 15-30ms
        solved = random.random() > 0.4  # 60%成功率
    
    return solved


def test_basic_usage():
    """测试基本使用"""
    print("=" * 60)
    print("测试1: 基本使用（仅控制台输出）")
    print("=" * 60)
    
    monitor = create_monitor(
        log_dir=None,  # 不保存日志
        print_interval=3.0,  # 3秒打印一次
        enable_logging=False,
        enable_console=True
    )
    
    print("开始模拟IK求解（30次）...\n")
    for i in range(30):
        start_time = time.perf_counter()
        solved = simulate_ik_solve('easy')
        elapsed = time.perf_counter() - start_time
        
        monitor.record_solve(elapsed, solved)
        time.sleep(0.1)  # 模拟10Hz频率
    
    print("\n最终统计:")
    stats = monitor.get_current_stats()
    print(f"  总尝试: {stats['total_attempts']}")
    print(f"  成功率: {stats['success_rate']:.1f}%")
    print(f"  平均耗时: {stats['avg_time_ms']:.2f} ms")
    
    monitor.close()
    print("\n" + "=" * 60 + "\n")


def test_with_logging():
    """测试带日志记录"""
    print("=" * 60)
    print("测试2: 带CSV日志记录")
    print("=" * 60)
    
    import os
    log_dir = "./test_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    monitor = create_monitor(
        log_dir=log_dir,
        print_interval=2.0,
        enable_logging=True,
        enable_console=True
    )
    
    print("开始模拟不同难度的IK求解（50次）...\n")
    difficulties = ['easy'] * 20 + ['medium'] * 20 + ['hard'] * 10
    
    for i, difficulty in enumerate(difficulties):
        start_time = time.perf_counter()
        solved = simulate_ik_solve(difficulty)
        elapsed = time.perf_counter() - start_time
        
        monitor.record_solve(elapsed, solved, iteration_count=random.randint(1, 50))
        
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/50")
        
        time.sleep(0.05)  # 模拟20Hz频率
    
    print("\n最终统计:")
    stats = monitor.get_current_stats()
    print(f"  总尝试: {stats['total_attempts']}")
    print(f"  成功: {stats['successful']}, 失败: {stats['failed']}")
    print(f"  成功率: {stats['success_rate']:.1f}%")
    print(f"  平均耗时: {stats['avg_time_ms']:.2f} ms")
    print(f"  最大耗时: {stats['max_time_ms']:.2f} ms")
    print(f"  最小耗时: {stats['min_time_ms']:.2f} ms")
    
    monitor.close()
    print("\n" + "=" * 60 + "\n")


def test_performance_warning():
    """测试性能警告"""
    print("=" * 60)
    print("测试3: 触发性能警告")
    print("=" * 60)
    
    monitor = create_monitor(
        log_dir=None,
        print_interval=2.0,
        enable_logging=False,
        enable_console=True
    )
    
    print("模拟低成功率和高耗时场景（20次）...\n")
    
    for i in range(20):
        start_time = time.perf_counter()
        # 故意制造低成功率和高耗时
        solved = simulate_ik_solve('hard')
        elapsed = time.perf_counter() - start_time + 0.015  # 额外增加15ms
        
        monitor.record_solve(elapsed, solved)
        time.sleep(0.1)
    
    print("\n最终统计:")
    stats = monitor.get_current_stats()
    print(f"  成功率: {stats['success_rate']:.1f}% (应该 < 80%, 触发警告)")
    print(f"  平均耗时: {stats['avg_time_ms']:.2f} ms (应该 > 20ms, 触发警告)")
    
    monitor.close()
    print("\n" + "=" * 60 + "\n")


def test_disabled_monitor():
    """测试禁用监控"""
    print("=" * 60)
    print("测试4: 监控器禁用（应该没有输出）")
    print("=" * 60)
    
    # 模拟 DEBUG_IK_PERF = False 的情况
    monitor = None
    
    print("开始模拟（监控器=None）...\n")
    for i in range(10):
        start_time = time.perf_counter()
        solved = simulate_ik_solve('easy')
        elapsed = time.perf_counter() - start_time
        
        # 只有当monitor不为None时才记录
        if monitor:
            monitor.record_solve(elapsed, solved)
        
        time.sleep(0.1)
    
    print("完成！没有监控输出（符合预期）")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "IK 性能监控模块测试" + " " * 28 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # 运行所有测试
    test_basic_usage()
    time.sleep(1)
    
    test_with_logging()
    time.sleep(1)
    
    test_performance_warning()
    time.sleep(1)
    
    test_disabled_monitor()
    
    print("✅ 所有测试完成！")
    print("\n提示: 查看 ./test_logs/ 目录查看生成的CSV日志文件")

