"""
诊断IK失败原因

对比添加工作空间约束前后的失败模式
"""

import pandas as pd
import numpy as np
import json
import sys

def analyze_failure_pattern(csv_path, json_path=None):
    """分析失败模式"""
    print("=" * 80)
    print("IK 失败原因诊断")
    print("=" * 80)
    print(f"\n📁 CSV数据: {csv_path}")
    if json_path:
        print(f"📁 JSON数据: {json_path}")
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    total = len(df)
    success = df[df['solved'] == 1]
    failure = df[df['solved'] == 0]
    
    success_count = len(success)
    failure_count = len(failure)
    
    print(f"\n📊 总体统计")
    print(f"  总尝试:   {total}")
    print(f"  成功:     {success_count} ({success_count/total*100:.2f}%)")
    print(f"  失败:     {failure_count} ({failure_count/total*100:.2f}%)")
    
    # 失败时间分布
    if failure_count > 0:
        print(f"\n⏱️  失败时间分布")
        
        # 按时间窗口统计
        df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        
        # 找到失败集中的时间段
        failure_times = df[df['solved'] == 0]['time_seconds'].values
        
        if len(failure_times) > 0:
            # 识别失败簇
            failure_clusters = []
            current_cluster = [failure_times[0]]
            
            for t in failure_times[1:]:
                if t - current_cluster[-1] < 2.0:  # 2秒内认为是同一簇
                    current_cluster.append(t)
                else:
                    if len(current_cluster) >= 3:  # 至少3次失败才算一个簇
                        failure_clusters.append(current_cluster)
                    current_cluster = [t]
            
            if len(current_cluster) >= 3:
                failure_clusters.append(current_cluster)
            
            print(f"  识别到 {len(failure_clusters)} 个失败簇:")
            for i, cluster in enumerate(failure_clusters[:5], 1):  # 只显示前5个
                start_time = cluster[0]
                end_time = cluster[-1]
                duration = end_time - start_time
                count = len(cluster)
                print(f"    簇 {i}: {start_time:.1f}s - {end_time:.1f}s (持续 {duration:.1f}s, {count}次失败)")
        
        # 分析失败耗时
        fail_times = failure['elapsed_ms']
        print(f"\n⏱️  失败求解耗时")
        print(f"  平均: {fail_times.mean():.2f} ms")
        print(f"  中位数: {fail_times.median():.2f} ms")
        print(f"  最大: {fail_times.max():.2f} ms")
        
        # 耗时分布
        slow_failures = len(fail_times[fail_times > 7])
        print(f"  耗时>7ms的失败: {slow_failures} ({slow_failures/failure_count*100:.1f}%)")
    
    # 如果有JSON数据（失败记录），分析目标位置
    if json_path:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            failures = data['data']
            if len(failures) > 0:
                print(f"\n🎯 失败时的目标位置分析 ({len(failures)}条记录)")
                
                # 提取位置
                left_positions = np.array([[f[5], f[6], f[7]] for f in failures])
                right_positions = np.array([[f[12], f[13], f[14]] for f in failures])
                
                # 假设肩部位置（需要从实际代码获取）
                left_shoulder = np.array([0.0, 0.2, 0.5])  # 示例值
                right_shoulder = np.array([0.0, -0.2, 0.5])  # 示例值
                
                left_distances = np.linalg.norm(left_positions - left_shoulder, axis=1)
                right_distances = np.linalg.norm(right_positions - right_shoulder, axis=1)
                
                max_reach = 0.436  # l1 + l2
                effective_reach = 0.420  # max_reach - safety_margin
                
                print(f"\n  左臂:")
                print(f"    平均距离: {left_distances.mean():.3f} m")
                print(f"    最大距离: {left_distances.max():.3f} m")
                print(f"    接近边界(>0.40m): {(left_distances > 0.40).sum()} ({(left_distances > 0.40).sum()/len(failures)*100:.1f}%)")
                print(f"    超出有效半径(>0.42m): {(left_distances > effective_reach).sum()}")
                
                print(f"\n  右臂:")
                print(f"    平均距离: {right_distances.mean():.3f} m")
                print(f"    最大距离: {right_distances.max():.3f} m")
                print(f"    接近边界(>0.40m): {(right_distances > 0.40).sum()} ({(right_distances > 0.40).sum()/len(failures)*100:.1f}%)")
                print(f"    超出有效半径(>0.42m): {(right_distances > effective_reach).sum()}")
                
                # 检查是否在边界上
                left_at_boundary = (left_distances >= 0.41) & (left_distances <= 0.421)
                right_at_boundary = (right_distances >= 0.41) & (right_distances <= 0.421)
                
                print(f"\n  🔍 边界失败分析:")
                print(f"    左臂在边界(0.41-0.421m): {left_at_boundary.sum()} ({left_at_boundary.sum()/len(failures)*100:.1f}%)")
                print(f"    右臂在边界(0.41-0.421m): {right_at_boundary.sum()} ({right_at_boundary.sum()/len(failures)*100:.1f}%)")
                
                if left_at_boundary.sum() > len(failures) * 0.5 or right_at_boundary.sum() > len(failures) * 0.5:
                    print(f"\n  ⚠️  警告: 超过50%的失败发生在工作空间边界!")
                    print(f"      可能原因: 约束后的边界位置容易触发关节限制")
                    print(f"      建议: 增大 safety_margin 到 0.020-0.025m")
        
        except Exception as e:
            print(f"\n⚠️  无法读取JSON: {e}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80 + "\n")


def compare_logs(csv_before, csv_after):
    """对比前后两个日志"""
    print("=" * 80)
    print("对比分析: 添加约束前 vs 添加约束后")
    print("=" * 80)
    
    df_before = pd.read_csv(csv_before)
    df_after = pd.read_csv(csv_after)
    
    before_total = len(df_before)
    before_success = (df_before['solved'] == 1).sum()
    before_fail = before_total - before_success
    before_rate = before_success / before_total * 100
    
    after_total = len(df_after)
    after_success = (df_after['solved'] == 1).sum()
    after_fail = after_total - after_success
    after_rate = after_success / after_total * 100
    
    print(f"\n📊 成功率对比")
    print(f"  添加约束前: {before_success}/{before_total} = {before_rate:.2f}%")
    print(f"  添加约束后: {after_success}/{after_total} = {after_rate:.2f}%")
    print(f"  变化: {after_rate - before_rate:+.2f} 个百分点")
    
    if after_rate < before_rate:
        print(f"\n  ⚠️  警告: 成功率下降！")
        print(f"  可能原因:")
        print(f"    1. 约束后的边界位置更容易触发关节限制")
        print(f"    2. safety_margin 太小，导致边界奇异")
        print(f"    3. 约束逻辑可能有问题")
    else:
        print(f"\n  ✅ 成功率提升!")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    log_dir = Path(__file__).parent / "log"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        json_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # 使用最新的日志
        csv_files = sorted(log_dir.glob("ik_performance_*.csv"), key=lambda p: p.stat().st_mtime)
        if len(csv_files) >= 1:
            csv_path = csv_files[-1]
            
            # 尝试找JSON
            csv_stem = csv_path.stem.replace('ik_performance_', 'teleop_')
            json_path = log_dir / f"{csv_stem}.json"
            if not json_path.exists():
                json_path = None
        else:
            print("❌ 未找到日志文件")
            sys.exit(1)
    
    # 分析失败模式
    analyze_failure_pattern(csv_path, json_path)
    
    # 如果有多个日志，进行对比
    csv_files = sorted(log_dir.glob("ik_performance_*.csv"), key=lambda p: p.stat().st_mtime)
    if len(csv_files) >= 2:
        print("\n")
        compare_logs(csv_files[-2], csv_files[-1])

