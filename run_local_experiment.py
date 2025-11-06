#!/usr/bin/env python3
"""
本地环境完全独立实验脚本
不依赖复杂的模块导入，直接运行
"""

import os
import sys
import time
import threading
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
import subprocess


def check_dependencies():
    """检查必要的依赖"""
    required_packages = ['pandas', 'numpy', 'psutil']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {missing_packages}")
        print("请运行: pip3 install " + " ".join(missing_packages))
        return False
    
    return True


class SimpleSystemMonitor:
    """简化的系统监控器"""
    
    def __init__(self, output_file, interval=1):
        self.output_file = output_file
        self.interval = interval
        self.running = False
        
    def start(self):
        """开始监控"""
        import psutil
        
        self.running = True
        
        # 创建CSV文件
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cpu_percent', 'memory_percent', 'disk_percent'])
        
        print(f"✅ 系统监控已启动: {self.output_file}")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                with open(self.output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, cpu_percent, memory_percent, disk_percent])
                
                print(f"[{timestamp}] CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%")
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"监控错误: {e}")
                break
    
    def stop(self):
        """停止监控"""
        self.running = False
        print("✅ 系统监控已停止")


class SimpleLogParser:
    """简化的日志解析器"""
    
    def __init__(self, log_file, output_file):
        self.log_file = log_file
        self.output_file = output_file
        self.running = False
        
    def start(self):
        """开始解析"""
        self.running = True
        
        # 创建CSV文件
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'loss', 'accuracy', 'learning_rate', 'step', 'epoch', 'event_type'])
        
        print(f"✅ 日志解析已启动: {self.output_file}")
        
        # 等待日志文件创建
        while self.running and not os.path.exists(self.log_file):
            time.sleep(0.5)
        
        if not self.running:
            return
            
        # 解析日志
        processed_lines = 0
        while self.running:
            try:
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        lines = f.readlines()
                    
                    # 处理新行
                    for line in lines[processed_lines:]:
                        self.parse_line(line.strip())
                        processed_lines += 1
                
                time.sleep(1)
                
            except Exception as e:
                print(f"日志解析错误: {e}")
                break
    
    def parse_line(self, line):
        """解析单行日志"""
        if not line:
            return
            
        try:
            # 提取时间戳
            timestamp_match = line.split(' - ')[0] if ' - ' in line else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 提取指标
            loss = None
            accuracy = None
            learning_rate = None
            step = None
            epoch = None
            event_type = "info"
            
            # 解析Loss
            if "Loss:" in line:
                import re
                loss_match = re.search(r'Loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|nan)', line)
                if loss_match:
                    loss_val = loss_match.group(1)
                    loss = float('nan') if loss_val == 'nan' else float(loss_val)
            
            # 解析Accuracy
            if "Accuracy:" in line:
                import re
                acc_match = re.search(r'Accuracy:\s*([0-9]*\.?[0-9]+)', line)
                if acc_match:
                    accuracy = float(acc_match.group(1))
            
            # 解析Learning Rate
            if "LR:" in line:
                import re
                lr_match = re.search(r'LR:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
                if lr_match:
                    learning_rate = float(lr_match.group(1))
            
            # 解析Step
            if "Step" in line:
                import re
                step_match = re.search(r'Step\s*(\d+)', line)
                if step_match:
                    step = int(step_match.group(1))
            
            # 解析Epoch
            if "Epoch" in line:
                import re
                epoch_match = re.search(r'Epoch\s*(\d+)', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
            
            # 检测事件类型
            if "ERROR" in line or "CRITICAL" in line:
                event_type = "error"
            elif "WARNING" in line:
                event_type = "warning"
            elif "nan" in line.lower():
                event_type = "anomaly"
            
            # 写入CSV
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_match, loss, accuracy, learning_rate, step, epoch, event_type])
                
        except Exception as e:
            print(f"解析行错误: {e}")
    
    def stop(self):
        """停止解析"""
        self.running = False
        print("✅ 日志解析已停止")


def generate_training_simulation():
    """生成训练模拟日志"""
    log_file = "local_training.log"
    start_time = datetime.now()
    
    print("🚀 开始生成训练模拟日志...")
    
    with open(log_file, 'w') as f:
        # 训练开始
        log_time = start_time
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - 开始BERT-IMDB微调训练"
        f.write(log_entry + '\n')
        f.flush()
        print(f"训练进度: 4% - {log_entry.split(' - ')[-1]}")
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=5)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - 模型: bert-base-uncased, 设备: CPU"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        # 正常训练阶段
        training_steps = [
            (10, 1, 1563, 0.693, 0.500, 2e-05),
            (15, 50, 1563, 0.620, 0.640, 1.98e-05),
            (20, 100, 1563, 0.580, 0.720, 1.96e-05),
            (25, 150, 1563, 0.540, 0.780, 1.94e-05),
            (30, 200, 1563, 0.500, 0.820, 1.92e-05),
        ]
        
        for offset, step, total_steps, loss, accuracy, lr in training_steps:
            log_time = start_time + timedelta(seconds=offset)
            log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Epoch 1/3, Step {step}/{total_steps}, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, LR: {lr:.2e}"
            f.write(log_entry + '\n')
            f.flush()
            print(f"训练进度: {20 + offset//2}% - {log_entry.split(' - ')[-1]}")
            time.sleep(1)
        
        # 故障注入点1: NaN Loss
        log_time = start_time + timedelta(seconds=35)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - 检测到学习率异常，Loss开始不稳定"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=36)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - Loss: nan, Accuracy: 0.000, LR: 1.90e-05"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=37)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - CRITICAL - NaN detected in loss computation"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=40)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - 训练已恢复, Loss: 0.520, Accuracy: 0.800, LR: 1.88e-05"
        f.write(log_entry + '\n')
        f.flush()
        print("训练进度: 45% - 训练已恢复")
        time.sleep(1)
        
        # 继续训练
        recovery_steps = [
            (45, 300, 1563, 0.480, 0.840, 1.86e-05),
            (50, 400, 1563, 0.450, 0.860, 1.84e-05),
        ]
        
        for offset, step, total_steps, loss, accuracy, lr in recovery_steps:
            log_time = start_time + timedelta(seconds=offset)
            log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Epoch 1/3, Step {step}/{total_steps}, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, LR: {lr:.2e}"
            f.write(log_entry + '\n')
            f.flush()
            time.sleep(1)
        
        # 故障注入点2: I/O瓶颈
        log_time = start_time + timedelta(seconds=60)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - 数据加载速度下降，检测到I/O瓶颈"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=65)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - 数据加载延迟: 5.2s (正常: 0.1s)"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=70)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - I/O瓶颈已解决，训练继续"
        f.write(log_entry + '\n')
        f.flush()
        print("训练进度: 60% - I/O瓶颈已解决")
        time.sleep(1)
        
        # 继续训练
        log_time = start_time + timedelta(seconds=75)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Epoch 1/3, Step 500/1563, Loss: 0.420, Accuracy: 0.880, LR: 1.82e-05"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=80)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Epoch 2/3, Step 1563/3126, Loss: 0.300, Accuracy: 0.940, LR: 1.76e-05"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        # 故障注入点3: 资源争用
        log_time = start_time + timedelta(seconds=110)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - 检测到CPU资源争用"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=115)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - CPU利用率异常: 95% (正常: 60%)"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=120)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - 资源争用已解决"
        f.write(log_entry + '\n')
        f.flush()
        print("训练进度: 80% - 资源争用已解决")
        time.sleep(1)
        
        # 训练完成
        log_time = start_time + timedelta(seconds=125)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Epoch 3/3, Step 3126/3126, Loss: 0.240, Accuracy: 0.970, LR: 1.70e-05"
        f.write(log_entry + '\n')
        f.flush()
        time.sleep(1)
        
        log_time = start_time + timedelta(seconds=130)
        log_entry = f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - 训练完成！最终验证准确率: 0.975"
        f.write(log_entry + '\n')
        f.flush()
        print("训练进度: 100% - 训练完成")
    
    print(f"✅ 训练模拟完成: {log_file}")
    return log_file, start_time


def aggregate_data_simple():
    """简单的数据聚合"""
    print("\n📊 开始数据聚合...")
    
    try:
        import pandas as pd
        
        # 读取数据文件
        system_df = pd.read_csv("local_system_metrics.csv")
        training_df = pd.read_csv("local_training_metrics.csv")
        
        # 转换时间戳
        system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])
        training_df['timestamp'] = pd.to_datetime(training_df['timestamp'])
        
        # 按时间戳合并
        merged_df = pd.merge(system_df, training_df, on='timestamp', how='outer')
        merged_df = merged_df.sort_values('timestamp')
        
        # 填充缺失值
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        
        # 添加异常标注
        merged_df['is_anomaly'] = 0
        
        # 标记异常时间段
        anomaly_periods = [
            ('2025-11-05 18:00:35', '2025-11-05 18:00:40'),  # NaN Loss
            ('2025-11-05 18:01:00', '2025-11-05 18:01:10'),  # I/O瓶颈
            ('2025-11-05 18:01:50', '2025-11-05 18:02:00'),  # 资源争用
        ]
        
        for start_time, end_time in anomaly_periods:
            mask = (merged_df['timestamp'] >= start_time) & (merged_df['timestamp'] <= end_time)
            merged_df.loc[mask, 'is_anomaly'] = 1
        
        # 保存聚合数据
        output_file = "local_aggregated_data.csv"
        merged_df.to_csv(output_file, index=False)
        
        anomaly_count = merged_df['is_anomaly'].sum()
        print(f"✅ 数据聚合完成: {len(merged_df)} 条记录, {anomaly_count} 个异常点")
        print(f"💾 聚合数据已保存: {output_file}")
        
        return merged_df
        
    except Exception as e:
        print(f"❌ 数据聚合失败: {e}")
        return None


def main():
    """主函数"""
    print("🚀 本地环境完全独立实验")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    try:
        # 创建输出目录
        os.makedirs("local_results", exist_ok=True)
        
        # 1. 启动监控
        print("\n📊 启动系统监控...")
        system_monitor = SimpleSystemMonitor("local_system_metrics.csv", interval=1)
        monitor_thread = threading.Thread(target=system_monitor.start)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 2. 启动日志解析
        log_parser = SimpleLogParser("local_training.log", "local_training_metrics.csv")
        parser_thread = threading.Thread(target=log_parser.start)
        parser_thread.daemon = True
        parser_thread.start()
        
        time.sleep(2)  # 等待监控启动
        
        # 3. 生成训练模拟
        print("\n🚀 开始训练模拟...")
        log_file = generate_training_simulation()
        
        # 4. 等待数据收集
        print("\n⏳ 等待数据收集完成...")
        time.sleep(5)
        
        # 5. 停止监控
        system_monitor.stop()
        log_parser.stop()
        
        time.sleep(2)  # 等待线程结束
        
        # 6. 数据聚合
        aggregated_data = aggregate_data_simple()
        
        # 7. 实验总结
        print("\n🎉 本地实验完成！")
        print("\n📁 生成的文件:")
        
        files = [
            "local_training.log",
            "local_system_metrics.csv",
            "local_training_metrics.csv",
            "local_aggregated_data.csv"
        ]
        
        total_size = 0
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                total_size += size
                print(f"  ✅ {file} ({size:,} bytes)")
            else:
                print(f"  ❌ {file} (未生成)")
        
        print(f"\n📊 总数据量: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
        
        if aggregated_data is not None:
            print(f"\n📈 数据质量报告:")
            print(f"  - 时间范围: {len(aggregated_data)} 个时间点")
            print(f"  - 特征数量: {len(aggregated_data.columns)} 个")
            anomaly_ratio = aggregated_data['is_anomaly'].sum() / len(aggregated_data) * 100
            print(f"  - 异常比例: {anomaly_ratio:.1f}%")
        
        print("\n💡 下一步建议:")
        print("1. 使用Google Colab运行GPU版本获得更好效果")
        print("2. 分析生成的local_aggregated_data.csv进行异常检测研究")
        print("3. 基于实验数据完善您的TSE-Matrix和MLH-AD框架")
        
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()