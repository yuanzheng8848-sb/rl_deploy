"""
IK性能日志分析脚本

分析 ik_performance CSV 和 teleop JSON 日志，提供优化建议
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_ik_performance(csv_path, json_path=None, output_dir=None):
    """
    分析IK性能日志
    
    Args:
        csv_path: ik_performance CSV文件路径
        json_path: teleop JSON文件路径（可选）
        output_dir: 输出图表目录（可选）
    """
    print("=" * 80)
    print("IK 性能分析报告")
    print("=" * 80)
    print(f"\n📁 数据来源: {csv_path}")
    
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 基本统计
    total_attempts = len(df)
    successful = df['solved'].sum()
    failed = total_attempts - successful
    success_rate = (successful / total_attempts) * 100
    
    print(f"\n📊 基本统计")
    print(f"  总尝试次数: {total_attempts}")
    print(f"  成功次数:   {successful} ({success_rate:.2f}%)")
    print(f"  失败次数:   {failed} ({100-success_rate:.2f}%)")
    
    # 耗时统计
    elapsed_times = df['elapsed_ms']
    print(f"\n⏱️  耗时统计（毫秒）")
    print(f"  平均耗时:   {elapsed_times.mean():.2f} ms")
    print(f"  中位数:     {elapsed_times.median():.2f} ms")
    print(f"  最小耗时:   {elapsed_times.min():.2f} ms")
    print(f"  最大耗时:   {elapsed_times.max():.2f} ms")
    print(f"  标准差:     {elapsed_times.std():.2f} ms")
    print(f"  95分位数:   {elapsed_times.quantile(0.95):.2f} ms")
    print(f"  99分位数:   {elapsed_times.quantile(0.99):.2f} ms")
    
    # 成功vs失败的耗时对比
    if failed > 0:
        success_times = df[df['solved'] == 1]['elapsed_ms']
        fail_times = df[df['solved'] == 0]['elapsed_ms']
        
        print(f"\n📈 成功 vs 失败对比")
        print(f"  成功平均耗时: {success_times.mean():.2f} ms")
        print(f"  失败平均耗时: {fail_times.mean():.2f} ms")
        print(f"  差异:         {fail_times.mean() - success_times.mean():+.2f} ms")
    
    # 频率统计
    duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    frequency = total_attempts / duration if duration > 0 else 0
    print(f"\n🔄 运行频率")
    print(f"  运行时长:   {duration:.1f} 秒")
    print(f"  平均频率:   {frequency:.1f} Hz")
    print(f"  理想周期:   {1000/frequency:.1f} ms")
    
    # 耗时分布统计
    print(f"\n📊 耗时分布")
    bins = [0, 2, 5, 10, 20, 50, float('inf')]
    labels = ['<2ms', '2-5ms', '5-10ms', '10-20ms', '20-50ms', '>50ms']
    df['time_bin'] = pd.cut(df['elapsed_ms'], bins=bins, labels=labels)
    distribution = df['time_bin'].value_counts().sort_index()
    for label, count in distribution.items():
        percentage = (count / total_attempts) * 100
        print(f"  {label:8s}: {count:5d} ({percentage:5.2f}%)")
    
    # 性能评估
    print(f"\n🎯 性能评估")
    avg_time = elapsed_times.mean()
    p95_time = elapsed_times.quantile(0.95)
    
    if avg_time < 5:
        print(f"  ✅ 平均耗时优秀 ({avg_time:.2f} ms < 5 ms)")
    elif avg_time < 10:
        print(f"  ✅ 平均耗时良好 ({avg_time:.2f} ms < 10 ms)")
    elif avg_time < 20:
        print(f"  ⚠️  平均耗时可接受 ({avg_time:.2f} ms < 20 ms)")
    else:
        print(f"  ❌ 平均耗时过长 ({avg_time:.2f} ms > 20 ms)")
    
    if p95_time < 10:
        print(f"  ✅ 95%耗时优秀 ({p95_time:.2f} ms < 10 ms)")
    elif p95_time < 20:
        print(f"  ✅ 95%耗时良好 ({p95_time:.2f} ms < 20 ms)")
    else:
        print(f"  ⚠️  95%耗时偏高 ({p95_time:.2f} ms > 20 ms)")
    
    if success_rate > 95:
        print(f"  ✅ 成功率优秀 ({success_rate:.2f}% > 95%)")
    elif success_rate > 90:
        print(f"  ✅ 成功率良好 ({success_rate:.2f}% > 90%)")
    elif success_rate > 80:
        print(f"  ⚠️  成功率可接受 ({success_rate:.2f}% > 80%)")
    else:
        print(f"  ❌ 成功率过低 ({success_rate:.2f}% < 80%)")
    
    # 优化建议
    print(f"\n💡 优化建议")
    
    if avg_time > 10:
        print(f"  🔧 建议增大 theta_step_size 到 0.15 以减少平均耗时")
    
    if success_rate < 90:
        print(f"  🔧 建议减小 theta_step_size 到 0.05 以提高成功率")
        print(f"  🔧 建议增加 max_iterations 到 80 以提高成功率")
    
    if p95_time > 20:
        print(f"  🔧 95%耗时偏高，考虑检查是否在工作空间边界频繁操作")
    
    if avg_time < 5 and success_rate > 95:
        print(f"  ✨ 当前配置已经很好！theta=0.1, steps=50 是合适的平衡点")
    
    # 分析失败日志
    if json_path and Path(json_path).exists():
        print(f"\n❌ 失败案例分析")
        with open(json_path, 'r') as f:
            failure_data = json.load(f)
        
        num_failures = len(failure_data['data'])
        print(f"  记录的失败次数: {num_failures}")
        
        if num_failures > 0:
            print(f"  失败率: {(num_failures/total_attempts)*100:.2f}%")
            
            # 分析失败时的目标位置
            failures = failure_data['data']
            if len(failures) > 0:
                left_positions = np.array([[f[5], f[6], f[7]] for f in failures])
                right_positions = np.array([[f[12], f[13], f[14]] for f in failures])
                
                left_distances = np.linalg.norm(left_positions, axis=1)
                right_distances = np.linalg.norm(right_positions, axis=1)
                
                print(f"  左臂失败时平均距离: {left_distances.mean():.3f} m")
                print(f"  右臂失败时平均距离: {right_distances.mean():.3f} m")
                print(f"  最大工作半径: 0.436 m (l1+l2)")
                
                # 检查是否超出工作空间
                left_out_of_reach = (left_distances > 0.436).sum()
                right_out_of_reach = (right_distances > 0.436).sum()
                
                if left_out_of_reach > 0:
                    print(f"  ⚠️  左臂有 {left_out_of_reach} 次失败可能因超出工作空间")
                if right_out_of_reach > 0:
                    print(f"  ⚠️  右臂有 {right_out_of_reach} 次失败可能因超出工作空间")
    
    # 绘图
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n📊 生成可视化图表...")
        
        # 1. 耗时时间序列
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 耗时趋势
        axes[0, 0].plot(df.index, df['elapsed_ms'], alpha=0.5, linewidth=0.5)
        axes[0, 0].axhline(y=elapsed_times.mean(), color='r', linestyle='--', label=f'Mean: {elapsed_times.mean():.2f}ms')
        axes[0, 0].axhline(y=elapsed_times.quantile(0.95), color='orange', linestyle='--', label=f'P95: {elapsed_times.quantile(0.95):.2f}ms')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('IK Solve Time (ms)')
        axes[0, 0].set_title('IK Solve Time Trend')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 耗时直方图
        axes[0, 1].hist(df['elapsed_ms'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=elapsed_times.mean(), color='r', linestyle='--', label=f'Mean: {elapsed_times.mean():.2f}ms')
        axes[0, 1].axvline(x=elapsed_times.median(), color='g', linestyle='--', label=f'Median: {elapsed_times.median():.2f}ms')
        axes[0, 1].set_xlabel('IK Solve Time (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('IK Solve Time Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 成功率滚动窗口
        window_size = 100
        df['success_rate_rolling'] = df['solved'].rolling(window=window_size).mean() * 100
        axes[1, 0].plot(df.index, df['success_rate_rolling'])
        axes[1, 0].axhline(y=success_rate, color='r', linestyle='--', label=f'Overall: {success_rate:.2f}%')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title(f'Success Rate (Rolling Window: {window_size})')
        axes[1, 0].set_ylim([0, 105])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 箱线图
        if failed > 0:
            data_for_box = [
                df[df['solved'] == 1]['elapsed_ms'],
                df[df['solved'] == 0]['elapsed_ms']
            ]
            axes[1, 1].boxplot(data_for_box, labels=['Success', 'Failure'])
        else:
            axes[1, 1].boxplot([df['elapsed_ms']], labels=['All'])
        axes[1, 1].set_ylabel('IK Solve Time (ms)')
        axes[1, 1].set_title('Time Distribution by Result')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 累积分布
        sorted_times = np.sort(df['elapsed_ms'])
        cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
        axes[2, 0].plot(sorted_times, cumulative)
        axes[2, 0].axhline(y=95, color='r', linestyle='--', alpha=0.5)
        axes[2, 0].axvline(x=elapsed_times.quantile(0.95), color='r', linestyle='--', 
                          label=f'P95: {elapsed_times.quantile(0.95):.2f}ms', alpha=0.5)
        axes[2, 0].set_xlabel('IK Solve Time (ms)')
        axes[2, 0].set_ylabel('Cumulative Percentage (%)')
        axes[2, 0].set_title('Cumulative Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 时间段分析
        df['time_bucket'] = pd.cut(df.index, bins=10)
        time_stats = df.groupby('time_bucket')['elapsed_ms'].agg(['mean', 'std'])
        axes[2, 1].errorbar(range(len(time_stats)), time_stats['mean'], yerr=time_stats['std'], 
                           marker='o', capsize=5)
        axes[2, 1].set_xlabel('Time Bucket')
        axes[2, 1].set_ylabel('Mean IK Solve Time (ms)')
        axes[2, 1].set_title('Performance Over Time')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'ik_performance_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 保存图表到: {plot_path}")
        plt.close()
    
    print(f"\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80 + "\n")
    
    return {
        'total_attempts': total_attempts,
        'success_rate': success_rate,
        'avg_time_ms': elapsed_times.mean(),
        'p95_time_ms': elapsed_times.quantile(0.95),
        'p99_time_ms': elapsed_times.quantile(0.99),
    }


if __name__ == "__main__":
    # 默认路径
    log_dir = Path(__file__).parent / "log"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 自动找到最新的日志文件
        csv_files = list(log_dir.glob("ik_performance_*.csv"))
        if not csv_files:
            print("❌ 未找到 ik_performance CSV 文件")
            sys.exit(1)
        csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    # 尝试找到对应的JSON文件
    csv_stem = Path(csv_path).stem.replace('ik_performance_', 'teleop_')
    json_path = Path(csv_path).parent / f"{csv_stem}.json"
    if not json_path.exists():
        json_path = None
    
    # 输出目录
    output_dir = Path(csv_path).parent / "analysis"
    
    # 执行分析
    results = analyze_ik_performance(csv_path, json_path, output_dir)

