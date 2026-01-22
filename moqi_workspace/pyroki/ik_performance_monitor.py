"""
IK性能监控模块

提供IK求解的性能监控、统计和日志记录功能
"""

import time
import os
from datetime import datetime


class IKPerformanceMonitor:
    """IK性能监控器"""
    
    def __init__(self, log_dir=None, print_interval=10.0, enable_logging=True, enable_console=True):
        """
        初始化性能监控器
        
        Args:
            log_dir: 日志保存目录
            print_interval: 控制台打印间隔（秒）
            enable_logging: 是否启用文件日志
            enable_console: 是否启用控制台输出
        """
        self.enable_logging = enable_logging
        self.enable_console = enable_console
        self.print_interval = print_interval
        
        # 统计数据
        self.stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0.0,
            'max_time': 0.0,
            'min_time': float('inf'),
            'last_print_time': time.time(),
        }
        
        # GUI显示组件（由外部设置）
        self.gui_displays = {
            'time': None,
            'success_rate': None,
            'total': None,
        }
        
        # 日志文件
        self.log_file = None
        if enable_logging and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(log_dir, f"ik_performance_{timestamp}.csv")
            self.log_file = open(log_path, 'w')
            self.log_file.write("timestamp,elapsed_ms,solved,iteration_count\n")
            self.log_file.flush()
            self.log_path = log_path
            print(f"📝 IK性能日志: {log_path}")
    
    def set_gui_displays(self, time_display=None, success_rate_display=None, total_display=None):
        """
        设置GUI显示组件
        
        Args:
            time_display: 时间显示组件
            success_rate_display: 成功率显示组件
            total_display: 总数显示组件
        """
        self.gui_displays['time'] = time_display
        self.gui_displays['success_rate'] = success_rate_display
        self.gui_displays['total'] = total_display
    
    def record_solve(self, elapsed_time, solved, iteration_count=-1):
        """
        记录一次IK求解
        
        Args:
            elapsed_time: 求解耗时（秒）
            solved: 是否成功求解
            iteration_count: 迭代次数（可选）
        """
        # 更新统计
        self.stats['total_attempts'] += 1
        self.stats['total_time'] += elapsed_time
        self.stats['max_time'] = max(self.stats['max_time'], elapsed_time)
        self.stats['min_time'] = min(self.stats['min_time'], elapsed_time)
        
        if solved:
            self.stats['successful'] += 1
        else:
            self.stats['failed'] += 1
        
        # 更新GUI显示
        self._update_gui_displays(elapsed_time)
        
        # 写入日志文件
        if self.enable_logging and self.log_file:
            timestamp = datetime.now().isoformat()
            self.log_file.write(f"{timestamp},{elapsed_time*1000:.4f},{int(solved)},{iteration_count}\n")
            self.log_file.flush()
        
        # 定期打印统计
        if self.enable_console:
            self._print_periodic_stats()
    
    def _update_gui_displays(self, last_elapsed):
        """更新GUI显示"""
        if self.gui_displays['time']:
            self.gui_displays['time'].value = f"{last_elapsed*1000:.2f} ms"
        
        if self.stats['total_attempts'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_attempts']) * 100
            
            if self.gui_displays['success_rate']:
                self.gui_displays['success_rate'].value = f"{success_rate:.1f}%"
            
            if self.gui_displays['total']:
                self.gui_displays['total'].value = (
                    f"{self.stats['total_attempts']} "
                    f"(成功:{self.stats['successful']}, 失败:{self.stats['failed']})"
                )
    
    def _print_periodic_stats(self):
        """定期打印统计信息"""
        current_time = time.time()
        if current_time - self.stats['last_print_time'] >= self.print_interval:
            if self.stats['total_attempts'] > 0:
                avg_time = self.stats['total_time'] / self.stats['total_attempts']
                success_rate = (self.stats['successful'] / self.stats['total_attempts']) * 100
                
                print("\n" + "="*60)
                print("📊 IK 性能统计 (最近 {:.1f}秒)".format(self.print_interval))
                print("="*60)
                print(f"  总尝试次数: {self.stats['total_attempts']}")
                print(f"  成功次数:   {self.stats['successful']} ({success_rate:.1f}%)")
                print(f"  失败次数:   {self.stats['failed']} ({100-success_rate:.1f}%)")
                print(f"  平均耗时:   {avg_time*1000:.2f} ms")
                print(f"  最大耗时:   {self.stats['max_time']*1000:.2f} ms")
                print(f"  最小耗时:   {self.stats['min_time']*1000:.2f} ms")
                
                # 性能警告
                warnings = []
                if avg_time > 0.02:  # 超过20ms
                    warnings.append("⚠️  平均耗时过长，可能影响实时性!")
                if success_rate < 80:
                    warnings.append("⚠️  成功率过低，考虑调整theta步长!")
                
                if warnings:
                    print("  " + "\n  ".join(warnings))
                
                print("="*60 + "\n")
                
                # 重置统计（滚动窗口）
                self.reset_stats()
    
    def reset_stats(self):
        """重置统计数据（保留GUI显示）"""
        self.stats['total_attempts'] = 0
        self.stats['successful'] = 0
        self.stats['failed'] = 0
        self.stats['total_time'] = 0.0
        self.stats['max_time'] = 0.0
        self.stats['min_time'] = float('inf')
        self.stats['last_print_time'] = time.time()
    
    def get_current_stats(self):
        """
        获取当前统计数据
        
        Returns:
            dict: 统计数据字典
        """
        if self.stats['total_attempts'] > 0:
            avg_time = self.stats['total_time'] / self.stats['total_attempts']
            success_rate = (self.stats['successful'] / self.stats['total_attempts']) * 100
        else:
            avg_time = 0
            success_rate = 0
        
        return {
            'total_attempts': self.stats['total_attempts'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'success_rate': success_rate,
            'avg_time_ms': avg_time * 1000,
            'max_time_ms': self.stats['max_time'] * 1000,
            'min_time_ms': self.stats['min_time'] * 1000 if self.stats['min_time'] != float('inf') else 0,
        }
    
    def close(self):
        """关闭监控器，保存日志"""
        if self.log_file:
            self.log_file.close()
            if self.enable_console:
                print(f"✅ IK性能日志已保存: {self.log_path}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()


class IKPerformanceTimer:
    """IK性能计时器上下文管理器"""
    
    def __init__(self, monitor, iteration_count=-1):
        """
        初始化计时器
        
        Args:
            monitor: IKPerformanceMonitor实例
            iteration_count: 迭代次数（可选）
        """
        self.monitor = monitor
        self.iteration_count = iteration_count
        self.start_time = None
        self.solved = False
    
    def __enter__(self):
        """开始计时"""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """结束计时并记录"""
        if self.monitor and self.start_time:
            elapsed = time.perf_counter() - self.start_time
            self.monitor.record_solve(elapsed, self.solved, self.iteration_count)
    
    def set_result(self, solved):
        """设置求解结果"""
        self.solved = solved


def create_gui_components(viser_server):
    """
    在Viser GUI中创建性能监控组件
    
    Args:
        viser_server: Viser服务器实例
        
    Returns:
        tuple: (time_display, success_rate_display, total_display)
    """
    viser_server.gui.add_markdown("---\n### 🔧 IK 性能监控")
    time_display = viser_server.gui.add_text("IK 耗时", initial_value="- ms", disabled=True)
    success_rate_display = viser_server.gui.add_text("成功率", initial_value="-%", disabled=True)
    total_display = viser_server.gui.add_text("总尝试次数", initial_value="0", disabled=True)
    
    return time_display, success_rate_display, total_display


# 便捷函数
def create_monitor(log_dir=None, print_interval=10.0, enable_logging=True, enable_console=True):
    """
    创建性能监控器的便捷函数
    
    Args:
        log_dir: 日志保存目录
        print_interval: 控制台打印间隔（秒）
        enable_logging: 是否启用文件日志
        enable_console: 是否启用控制台输出
        
    Returns:
        IKPerformanceMonitor: 监控器实例
    """
    return IKPerformanceMonitor(
        log_dir=log_dir,
        print_interval=print_interval,
        enable_logging=enable_logging,
        enable_console=enable_console
    )
