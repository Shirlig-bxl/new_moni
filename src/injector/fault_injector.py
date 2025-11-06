"""
故障注入控制器
负责在训练过程中主动触发各种故障
"""

import os
import time
import signal
import subprocess
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import psutil
import argparse


class FaultInjector:
    """故障注入器类"""
    
    def __init__(self, enable_training_hooks: bool = True):
        self.running = False
        self.injection_thread = None
        self.fault_schedule = []  # 故障注入计划
        self.start_time = None
        self.enable_training_hooks = enable_training_hooks
        self.training_hooks = []
        
        # 如果启用了训练钩子，导入相关模块
        if enable_training_hooks:
            try:
                from .training_hook import create_fault_hooks_from_config
                self.create_fault_hooks_from_config = create_fault_hooks_from_config
            except ImportError:
                self.enable_training_hooks = False
                print("警告: 无法导入训练钩子模块，训练钩子功能已禁用")
        
        print("故障注入器初始化完成")
    
    def create_training_hooks(self, config: Dict[str, Any]) -> List[Any]:
        """
        从配置创建训练故障注入钩子
        
        Args:
            config: 故障注入配置
            
        Returns:
            故障注入钩子列表
        """
        if not self.enable_training_hooks:
            print("警告: 训练钩子功能已禁用")
            return []
        
        try:
            return self.create_fault_hooks_from_config(config)
        except Exception as e:
            print(f"创建训练钩子失败: {e}")
            return []
    
    def get_training_fault_injector(self, config: Dict[str, Any]) -> Any:
        """
        获取训练故障注入器回调
        
        Args:
            config: 故障注入配置
            
        Returns:
            TrainingFaultInjector实例
        """
        if not self.enable_training_hooks:
            print("警告: 训练钩子功能已禁用")
            return None
        
        try:
            from .training_hook import TrainingFaultInjector
            fault_injector = TrainingFaultInjector()
            hooks = self.create_training_hooks(config)
            for hook in hooks:
                fault_injector.add_hook(hook)
            print(f"已创建训练故障注入器，包含 {len(hooks)} 个钩子")
            return fault_injector
        except Exception as e:
            print(f"获取训练故障注入器失败: {e}")
            return None
    
    def schedule_fault(self, fault_type: str, delay: float, duration: float = 0, **kwargs):
        """
        调度故障注入
        
        Args:
            fault_type: 故障类型 ("io_stress", "resource_competition", "process_kill", "nan_loss", "oom", "non_convergence")
            delay: 延迟时间（秒）
            duration: 持续时间（秒），0表示一次性操作
            **kwargs: 故障特定参数
        """
        fault_config = {
            'fault_type': fault_type,
            'delay': delay,
            'duration': duration,
            'params': kwargs,
            'executed': False
        }
        
        self.fault_schedule.append(fault_config)
        print(f"已调度故障: {fault_type}, 延迟: {delay}s, 持续: {duration}s")
    
    def inject_io_stress(self, target_dir: str = "/tmp", duration: float = 60, 
                        read_rate: str = "100M", write_rate: str = "100M"):
        """
        注入I/O压力
        
        Args:
            target_dir: 目标目录
            duration: 持续时间（秒）
            read_rate: 读取速率
            write_rate: 写入速率
        """
        print(f"[FAULT INJECTION] 开始I/O压力测试: {target_dir}, 持续 {duration}s")
        
        try:
            # 使用dd命令创建I/O压力
            stress_processes = []
            
            # 写入压力
            write_cmd = [
                'dd', 'if=/dev/zero', f'of={target_dir}/stress_write_test',
                f'bs=1M', 'count=1000', 'oflag=direct'
            ]
            
            # 读取压力 (如果文件存在)
            read_cmd = [
                'dd', f'if={target_dir}/stress_write_test', 'of=/dev/null',
                f'bs=1M', 'iflag=direct'
            ]
            
            # 启动写入压力
            write_proc = subprocess.Popen(write_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stress_processes.append(write_proc)
            
            # 等待一段时间后启动读取压力
            time.sleep(2)
            
            # 启动多个读取进程
            for i in range(3):
                try:
                    read_proc = subprocess.Popen(read_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    stress_processes.append(read_proc)
                except:
                    pass
            
            # 等待指定时间
            time.sleep(duration)
            
            # 终止所有压力进程
            for proc in stress_processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            
            # 清理测试文件
            try:
                os.remove(f'{target_dir}/stress_write_test')
            except:
                pass
            
            print(f"[FAULT INJECTION] I/O压力测试结束")
            
        except Exception as e:
            print(f"[FAULT INJECTION] I/O压力测试失败: {e}")
    
    def inject_resource_competition(self, target_process_name: str = "python", 
                                  duration: float = 60, competitor_type: str = "gpu"):
        """
        注入资源竞争
        
        Args:
            target_process_name: 目标进程名称
            duration: 持续时间（秒）
            competitor_type: 竞争类型 ("gpu", "cpu", "memory")
        """
        print(f"[FAULT INJECTION] 开始资源竞争: {competitor_type}, 持续 {duration}s")
        
        competitor_processes = []
        
        try:
            if competitor_type == "gpu":
                # GPU竞争：启动GPU密集型任务
                gpu_stress_cmd = [
                    'python', '-c',
                    '''
import torch
import time
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    # 创建大矩阵进行计算
    for i in range(1000):
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        time.sleep(0.1)
else:
    print("CUDA not available")
                    '''
                ]
                
                # 启动多个GPU竞争进程
                for i in range(2):
                    try:
                        proc = subprocess.Popen(gpu_stress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        competitor_processes.append(proc)
                    except Exception as e:
                        print(f"启动GPU竞争进程失败: {e}")
            
            elif competitor_type == "cpu":
                # CPU竞争：启动CPU密集型任务
                cpu_stress_cmd = [
                    'python', '-c',
                    '''
import time
import multiprocessing
def cpu_intensive():
    while True:
        sum(i*i for i in range(10000))

processes = []
for i in range(multiprocessing.cpu_count()):
    p = multiprocessing.Process(target=cpu_intensive)
    p.start()
    processes.append(p)

time.sleep(60)
for p in processes:
    p.terminate()
                    '''
                ]
                
                proc = subprocess.Popen(cpu_stress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                competitor_processes.append(proc)
            
            elif competitor_type == "memory":
                # 内存竞争：分配大量内存
                memory_stress_cmd = [
                    'python', '-c',
                    '''
import time
import numpy as np
arrays = []
try:
    for i in range(10):
        # 分配1GB内存
        arr = np.zeros((1024, 1024, 128), dtype=np.float64)
        arrays.append(arr)
        time.sleep(1)
    time.sleep(60)
except MemoryError:
    print("Memory allocation failed")
                    '''
                ]
                
                proc = subprocess.Popen(memory_stress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                competitor_processes.append(proc)
            
            # 等待指定时间
            time.sleep(duration)
            
        except Exception as e:
            print(f"[FAULT INJECTION] 资源竞争注入失败: {e}")
        
        finally:
            # 终止所有竞争进程
            for proc in competitor_processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            
            print(f"[FAULT INJECTION] 资源竞争结束")
    
    def inject_process_kill(self, target_process_name: str = "python", 
                           target_cmdline_pattern: str = "train.py"):
        """
        注入进程终止
        
        Args:
            target_process_name: 目标进程名称
            target_cmdline_pattern: 命令行模式匹配
        """
        print(f"[FAULT INJECTION] 查找并终止进程: {target_process_name}, 模式: {target_cmdline_pattern}")
        
        killed_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if (proc_info['name'] and target_process_name in proc_info['name'].lower() and
                        proc_info['cmdline'] and any(target_cmdline_pattern in cmd for cmd in proc_info['cmdline'])):
                        
                        print(f"[FAULT INJECTION] 找到目标进程: PID={proc.pid}, CMD={' '.join(proc_info['cmdline'])}")
                        
                        # 发送SIGTERM信号
                        proc.terminate()
                        killed_processes.append(proc.pid)
                        
                        # 等待3秒，如果还没结束就强制杀死
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                            print(f"[FAULT INJECTION] 强制杀死进程: PID={proc.pid}")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            if killed_processes:
                print(f"[FAULT INJECTION] 已终止进程: {killed_processes}")
            else:
                print(f"[FAULT INJECTION] 未找到匹配的进程")
                
        except Exception as e:
            print(f"[FAULT INJECTION] 进程终止失败: {e}")
    
    def inject_nan_loss(self, target_file: str = None, corruption_probability: float = 0.5):
        """
        注入NaN Loss故障 - 通过修改训练脚本或配置文件
        
        Args:
            target_file: 目标训练脚本文件路径
            corruption_probability: 损坏概率
        """
        print(f"[FAULT INJECTION] 开始NaN Loss注入: 损坏概率={corruption_probability}")
        
        try:
            # 方法1: 创建一个临时的故障配置文件
            fault_config_content = f"""
# 临时故障注入配置 - NaN Loss
import torch
import numpy as np

# 原始forward函数的包装器
def inject_nan_to_loss(original_loss):
    if torch.rand(1).item() < {corruption_probability}:
        print("[FAULT INJECTION] 注入NaN到损失函数")
        return torch.tensor(float('nan'), requires_grad=True)
    return original_loss

# 梯度损坏函数
def corrupt_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.rand(1).item() < {corruption_probability}:
            param.grad = torch.full_like(param.grad, float('nan'))
            print(f"[FAULT INJECTION] 已损坏 {{name}} 的梯度")
"""
            
            # 写入临时故障注入文件
            fault_file = "/tmp/nan_loss_injection.py"
            with open(fault_file, 'w') as f:
                f.write(fault_config_content)
            
            print(f"[FAULT INJECTION] NaN Loss注入配置已写入: {fault_file}")
            print("[FAULT INJECTION] 注意: 需要在训练脚本中手动导入此配置以激活故障")
            
        except Exception as e:
            print(f"[FAULT INJECTION] NaN Loss注入失败: {e}")
    
    def inject_oom(self, memory_size_mb: int = 1024, allocation_pattern: str = "exponential"):
        """
        注入OOM (内存溢出) 故障
        
        Args:
            memory_size_mb: 要分配的内存大小(MB)
            allocation_pattern: 分配模式 ("exponential", "linear")
        """
        print(f"[FAULT INJECTION] 开始OOM注入: 分配 {memory_size_mb}MB 内存")
        
        allocated_arrays = []
        
        try:
            if allocation_pattern == "exponential":
                # 指数增长分配
                current_size = 1  # 1MB起始
                total_allocated = 0
                
                while total_allocated < memory_size_mb:
                    size_mb = min(current_size, memory_size_mb - total_allocated)
                    if size_mb <= 0:
                        break
                    
                    # 分配内存 (每MB约250k个float32)
                    array_size = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
                    array = [0.0] * array_size
                    allocated_arrays.append(array)
                    
                    total_allocated += size_mb
                    current_size *= 2
                    
                    print(f"[FAULT INJECTION] 已分配 {size_mb}MB, 总计: {total_allocated}MB")
                    time.sleep(0.1)  # 短暂延迟
                    
            else:  # linear allocation
                # 线性分配
                chunk_size = 64  # 64MB chunks
                chunks_needed = memory_size_mb // chunk_size
                remainder = memory_size_mb % chunk_size
                
                for i in range(chunks_needed):
                    array_size = int(chunk_size * 1024 * 1024 / 4)
                    array = [0.0] * array_size
                    allocated_arrays.append(array)
                    print(f"[FAULT INJECTION] 已分配块 {i+1}/{chunks_needed} ({chunk_size}MB)")
                    time.sleep(0.1)
                
                # 分配剩余内存
                if remainder > 0:
                    array_size = int(remainder * 1024 * 1024 / 4)
                    array = [0.0] * array_size
                    allocated_arrays.append(array)
                    print(f"[FAULT INJECTION] 已分配剩余 {remainder}MB")
            
            print(f"[FAULT INJECTION] OOM注入完成，共分配 {len(allocated_arrays)} 个内存块")
            
            # 保持内存分配一段时间
            time.sleep(30)  # 保持30秒
            
        except MemoryError as e:
            print(f"[FAULT INJECTION] OOM注入成功触发内存错误: {e}")
        except Exception as e:
            print(f"[FAULT INJECTION] OOM注入失败: {e}")
        finally:
            # 清理分配的内存
            allocated_arrays.clear()
            print("[FAULT INJECTION] 已清理分配的内存")
    
    def inject_non_convergence(self, lr_multiplier: float = 1000.0, 
                              corruption_type: str = "too_high", duration: float = 60):
        """
        注入不收敛故障 - 通过创建干扰配置
        
        Args:
            lr_multiplier: 学习率倍数
            corruption_type: 损坏类型 ("too_high", "too_low")
            duration: 持续时间（秒）
        """
        print(f"[FAULT INJECTION] 开始不收敛注入: {corruption_type}, 倍数={lr_multiplier}")
        
        try:
            # 创建学习率干扰配置
            if corruption_type == "too_high":
                corrupted_lr = 2e-5 * lr_multiplier  # 基础学习率 * 倍数
                message = f"极高学习率: {corrupted_lr}"
            else:  # too_low
                corrupted_lr = 2e-5 / lr_multiplier  # 基础学习率 / 倍数
                message = f"极低学习率: {corrupted_lr}"
            
            # 创建干扰配置文件
            config_content = f"""
# 不收敛故障注入配置
# {message}

import json
import os

# 修改训练配置
def corrupt_training_config():
    config = {{
        "learning_rate": {corrupted_lr},
        "fault_type": "non_convergence",
        "corruption_type": "{corruption_type}",
        "original_lr": 2e-5,
        "multiplier": {lr_multiplier}
    }}
    
    with open("/tmp/corrupted_training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[FAULT INJECTION] 已创建损坏的训练配置: {{config['learning_rate']}}")
    return config

if __name__ == "__main__":
    corrupt_training_config()
"""
            
            # 写入配置文件
            config_file = "/tmp/non_convergence_injection.py"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # 执行配置生成
            subprocess.run(['python', config_file], check=True)
            
            print(f"[FAULT INJECTION] 不收敛注入配置已生成: {config_file}")
            print(f"[FAULT INJECTION] {message}")
            
            # 保持配置一段时间
            time.sleep(duration)
            
        except Exception as e:
            print(f"[FAULT INJECTION] 不收敛注入失败: {e}")
        finally:
            # 清理临时文件
            try:
                os.remove("/tmp/non_convergence_injection.py")
                os.remove("/tmp/corrupted_training_config.json")
                print("[FAULT INJECTION] 已清理临时配置文件")
            except:
                pass
    
    def execute_fault(self, fault_config: Dict[str, Any]):
        """执行单个故障注入"""
        fault_type = fault_config['fault_type']
        params = fault_config['params']
        
        print(f"[FAULT INJECTION] 执行故障: {fault_type}")
        
        try:
            if fault_type == "io_stress":
                self.inject_io_stress(**params)
            elif fault_type == "resource_competition":
                self.inject_resource_competition(**params)
            elif fault_type == "process_kill":
                self.inject_process_kill(**params)
            elif fault_type == "nan_loss":
                self.inject_nan_loss(**params)
            elif fault_type == "oom":
                self.inject_oom(**params)
            elif fault_type == "non_convergence":
                self.inject_non_convergence(**params)
            else:
                print(f"[FAULT INJECTION] 未知故障类型: {fault_type}")
        
        except Exception as e:
            print(f"[FAULT INJECTION] 故障执行失败: {e}")
    
    def injection_loop(self):
        """故障注入循环"""
        print("故障注入器开始运行")
        self.start_time = time.time()
        
        while self.running:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 检查是否有需要执行的故障
            for fault_config in self.fault_schedule:
                if not fault_config['executed'] and elapsed_time >= fault_config['delay']:
                    fault_config['executed'] = True
                    
                    # 在新线程中执行故障注入
                    fault_thread = threading.Thread(
                        target=self.execute_fault,
                        args=(fault_config,)
                    )
                    fault_thread.daemon = True
                    fault_thread.start()
            
            time.sleep(1)  # 每秒检查一次
        
        print("故障注入器停止运行")
    
    def start(self):
        """启动故障注入器"""
        if self.running:
            print("故障注入器已在运行中")
            return
        
        if not self.fault_schedule:
            print("没有调度的故障，故障注入器不会启动")
            return
        
        self.running = True
        self.injection_thread = threading.Thread(target=self.injection_loop)
        self.injection_thread.daemon = True
        self.injection_thread.start()
        print("故障注入器已启动")
    
    def stop(self):
        """停止故障注入器"""
        if not self.running:
            print("故障注入器未在运行")
            return
        
        self.running = False
        if self.injection_thread:
            self.injection_thread.join(timeout=5)
        print("故障注入器已停止")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_fault_schedule_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从配置创建故障调度"""
    schedule = []
    
    for fault_name, fault_config in config.items():
        schedule.append({
            'fault_type': fault_config['type'],
            'delay': fault_config['delay'],
            'duration': fault_config.get('duration', 0),
            'params': fault_config.get('params', {}),
            'executed': False
        })
    
    return schedule


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="故障注入工具")
    parser.add_argument("--fault-type", choices=["io_stress", "resource_competition", "process_kill", "nan_loss", "oom", "non_convergence"], 
                       help="故障类型")
    parser.add_argument("--delay", type=float, default=0, help="延迟时间（秒）")
    parser.add_argument("--duration", type=float, default=60, help="持续时间（秒）")
    parser.add_argument("--target-dir", default="/tmp", help="I/O压力目标目录")
    parser.add_argument("--competitor-type", choices=["gpu", "cpu", "memory"], default="gpu", 
                       help="资源竞争类型")
    parser.add_argument("--target-process", default="python", help="目标进程名称")
    parser.add_argument("--target-pattern", default="train.py", help="目标进程命令行模式")
    
    args = parser.parse_args()
    
    if not args.fault_type:
        print("请指定故障类型")
        return
    
    injector = FaultInjector()
    
    # 根据参数调度故障
    if args.fault_type == "io_stress":
        injector.schedule_fault(
            "io_stress", 
            delay=args.delay, 
            duration=args.duration,
            target_dir=args.target_dir
        )
    elif args.fault_type == "resource_competition":
        injector.schedule_fault(
            "resource_competition",
            delay=args.delay,
            duration=args.duration,
            competitor_type=args.competitor_type
        )
    elif args.fault_type == "process_kill":
        injector.schedule_fault(
            "process_kill",
            delay=args.delay,
            target_process_name=args.target_process,
            target_cmdline_pattern=args.target_pattern
        )
    elif args.fault_type == "nan_loss":
        injector.schedule_fault(
            "nan_loss",
            delay=args.delay,
            duration=args.duration,
            corruption_probability=0.5
        )
    elif args.fault_type == "oom":
        injector.schedule_fault(
            "oom",
            delay=args.delay,
            duration=args.duration,
            memory_size_mb=1024,
            allocation_pattern="exponential"
        )
    elif args.fault_type == "non_convergence":
        injector.schedule_fault(
            "non_convergence",
            delay=args.delay,
            duration=args.duration,
            lr_multiplier=1000.0,
            corruption_type="too_high"
        )
    
    try:
        injector.start()
        print("按 Ctrl+C 停止故障注入器")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        injector.stop()


if __name__ == "__main__":
    main()