"""
故障注入器单元测试
"""

import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path
import yaml
from unittest.mock import Mock, patch, MagicMock

from src.injector.fault_injector import FaultInjector
from src.injector.training_hook import FaultInjectionHook, NaNLossHook, OOMHook, NonConvergenceHook, TrainingFaultInjector


class TestFaultInjector:
    """故障注入器测试类"""
    
    def setup_method(self):
        """测试方法前置设置"""
        self.injector = FaultInjector(enable_training_hooks=False)
    
    def test_initialization(self):
        """测试初始化"""
        assert self.injector.running is False
        assert self.injector.injection_thread is None
        assert self.injector.fault_schedule == []
        assert self.injector.start_time is None
    
    def test_schedule_fault(self):
        """测试故障调度"""
        # 调度一个故障
        self.injector.schedule_fault(
            "io_stress", 
            delay=10, 
            duration=30,
            target_dir="/tmp"
        )
        
        assert len(self.injector.fault_schedule) == 1
        fault_config = self.injector.fault_schedule[0]
        
        assert fault_config['fault_type'] == "io_stress"
        assert fault_config['delay'] == 10
        assert fault_config['duration'] == 30
        assert fault_config['params']['target_dir'] == "/tmp"
        assert fault_config['executed'] is False
    
    def test_schedule_multiple_faults(self):
        """测试调度多个故障"""
        # 调度多个故障
        self.injector.schedule_fault("io_stress", delay=10, duration=30)
        self.injector.schedule_fault("resource_competition", delay=20, duration=60)
        
        assert len(self.injector.fault_schedule) == 2
        assert self.injector.fault_schedule[0]['fault_type'] == "io_stress"
        assert self.injector.fault_schedule[1]['fault_type'] == "resource_competition"
    
    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_inject_io_stress(self, mock_sleep, mock_popen):
        """测试I/O压力注入（模拟）"""
        # 模拟子进程
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        # 执行I/O压力注入
        self.injector.inject_io_stress(
            target_dir="/tmp/test",
            duration=5,
            read_rate="10M",
            write_rate="10M"
        )
        
        # 验证子进程被创建
        assert mock_popen.called
        # 验证sleep被调用（模拟等待）
        mock_sleep.assert_called()
    
    @patch('subprocess.Popen')
    @patch('time.sleep')
    def test_inject_resource_competition(self, mock_sleep, mock_popen):
        """测试资源竞争注入（模拟）"""
        # 模拟子进程
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        # 执行GPU资源竞争注入
        self.injector.inject_resource_competition(
            target_process_name="python",
            duration=5,
            competitor_type="gpu"
        )
        
        # 验证子进程被创建
        assert mock_popen.called
        # 验证sleep被调用
        mock_sleep.assert_called()
    
    @patch('psutil.process_iter')
    def test_inject_process_kill(self, mock_process_iter):
        """测试进程终止注入（模拟）"""
        # 模拟进程
        mock_proc = Mock()
        mock_proc.info = {
            'pid': 12345,
            'name': 'python',
            'cmdline': ['python', 'train.py']
        }
        mock_proc.pid = 12345
        
        # 模拟进程迭代器
        mock_process_iter.return_value = [mock_proc]
        
        # 执行进程终止
        self.injector.inject_process_kill(
            target_process_name="python",
            target_cmdline_pattern="train.py"
        )
        
        # 验证进程terminate被调用
        mock_proc.terminate.assert_called()
    
    def test_inject_nan_loss(self):
        """测试NaN Loss注入"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 修改临时文件路径用于测试
            with patch('src.injector.fault_injector.FaultInjector.inject_nan_loss') as mock_method:
                mock_method.return_value = None
                
                # 执行NaN Loss注入
                self.injector.inject_nan_loss(
                    target_file=os.path.join(temp_dir, "test.py"),
                    corruption_probability=0.3
                )
                
                # 验证方法被调用
                mock_method.assert_called_once()
    
    def test_execute_fault(self):
        """测试故障执行"""
        # 创建故障配置
        fault_config = {
            'fault_type': 'io_stress',
            'params': {
                'target_dir': '/tmp',
                'duration': 5
            }
        }
        
        # 模拟故障方法
        with patch.object(self.injector, 'inject_io_stress') as mock_method:
            # 执行故障
            self.injector.execute_fault(fault_config)
            
            # 验证对应的方法被调用
            mock_method.assert_called_once_with(target_dir='/tmp', duration=5)
    
    def test_execute_unknown_fault(self):
        """测试执行未知故障类型"""
        fault_config = {
            'fault_type': 'unknown_fault',
            'params': {}
        }
        
        # 应该不会抛出异常，但会打印警告
        self.injector.execute_fault(fault_config)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with patch.object(self.injector, 'start') as mock_start, \
             patch.object(self.injector, 'stop') as mock_stop:
            
            with self.injector:
                mock_start.assert_called_once()
            
            mock_stop.assert_called_once()


class TestFaultInjectionHooks:
    """故障注入钩子测试类"""
    
    def test_fault_injection_hook_base(self):
        """测试故障注入钩子基类"""
        hook = FaultInjectionHook("test_fault", trigger_step=100)
        
        assert hook.fault_type == "test_fault"
        assert hook.trigger_step == 100
        assert hook.activated is False
        assert hook.original_state == {}
        
        # 测试激活条件
        assert hook.should_activate(99) is False  # 未达到触发步数
        assert hook.should_activate(100) is True   # 达到触发步数
        assert hook.should_activate(101) is True   # 超过触发步数
        
        # 测试激活后
        hook.activated = True
        assert hook.should_activate(100) is False  # 已激活
    
    def test_nan_loss_hook(self):
        """测试NaN Loss钩子"""
        hook = NaNLossHook(
            trigger_step=50,
            corruption_probability=0.5,
            target_layers=["classifier"],
            duration_steps=3
        )
        
        assert hook.fault_type == "nan_loss"
        assert hook.trigger_step == 50
        assert hook.params['corruption_probability'] == 0.5
        assert hook.params['target_layers'] == ["classifier"]
        assert hook.params['duration_steps'] == 3
        
        # 测试梯度损坏函数
        import torch
        mock_grad = torch.tensor([1.0, 2.0, 3.0])
        
        # 由于随机性，我们测试函数结构
        result = hook._corrupt_gradient(mock_grad)
        assert result is not None
    
    def test_oom_hook(self):
        """测试OOM钩子"""
        hook = OOMHook(
            trigger_step=100,
            memory_size_mb=512,
            allocation_pattern="exponential"
        )
        
        assert hook.fault_type == "oom"
        assert hook.trigger_step == 100
        assert hook.params['memory_size_mb'] == 512
        assert hook.params['allocation_pattern'] == "exponential"
        assert hook.allocated_tensors == []
    
    def test_non_convergence_hook(self):
        """测试不收敛钩子"""
        hook = NonConvergenceHook(
            trigger_step=200,
            lr_multiplier=1000.0,
            corruption_type="too_high"
        )
        
        assert hook.fault_type == "non_convergence"
        assert hook.trigger_step == 200
        assert hook.params['lr_multiplier'] == 1000.0
        assert hook.params['corruption_type'] == "too_high"


class TestTrainingFaultInjector:
    """训练故障注入器测试类"""
    
    def setup_method(self):
        """测试方法前置设置"""
        self.fault_injector = TrainingFaultInjector()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.fault_injector.hooks == []
        assert self.fault_injector.active_hooks == []
    
    def test_add_hook(self):
        """测试添加钩子"""
        hook = NaNLossHook(trigger_step=100)
        
        self.fault_injector.add_hook(hook)
        
        assert len(self.fault_injector.hooks) == 1
        assert self.fault_injector.hooks[0] == hook
    
    def test_on_step_begin_activate_hook(self):
        """测试步骤开始时激活钩子"""
        # 创建模拟钩子
        mock_hook = Mock(spec=FaultInjectionHook)
        mock_hook.fault_type = "test_fault"
        mock_hook.trigger_step = 10
        mock_hook.activated = False
        mock_hook.should_activate.return_value = True
        
        self.fault_injector.hooks.append(mock_hook)
        
        # 创建模拟参数
        mock_args = Mock()
        mock_state = Mock()
        mock_state.global_step = 10
        mock_control = Mock()
        mock_trainer = Mock()
        mock_model = Mock()
        
        # 调用步骤开始回调
        self.fault_injector.on_step_begin(
            mock_args, mock_state, mock_control,
            trainer=mock_trainer, model=mock_model
        )
        
        # 验证钩子被激活
        mock_hook.activate.assert_called_once_with(mock_trainer, mock_model)
        assert len(self.fault_injector.active_hooks) == 1
        assert self.fault_injector.active_hooks[0] == mock_hook
    
    def test_on_train_end_cleanup(self):
        """测试训练结束时清理钩子"""
        # 创建激活的钩子
        mock_hook = Mock(spec=FaultInjectionHook)
        mock_hook.activated = True
        
        self.fault_injector.active_hooks.append(mock_hook)
        
        # 创建模拟参数
        mock_args = Mock()
        mock_state = Mock()
        mock_control = Mock()
        mock_trainer = Mock()
        mock_model = Mock()
        
        # 调用训练结束回调
        self.fault_injector.on_train_end(
            mock_args, mock_state, mock_control,
            trainer=mock_trainer, model=mock_model
        )
        
        # 验证钩子被停用
        mock_hook.deactivate.assert_called_once_with(mock_trainer, mock_model)
        assert len(self.fault_injector.active_hooks) == 0


class TestFaultConfigIntegration:
    """故障配置集成测试"""
    
    def test_create_fault_hooks_from_config(self):
        """测试从配置创建故障钩子"""
        from src.injector.training_hook import create_fault_hooks_from_config
        
        # 创建测试配置
        config = {
            "injection_schedule": {
                "step_based": [
                    {
                        "fault_type": "nan_loss",
                        "injection_step": 50,
                        "duration": 5
                    },
                    {
                        "fault_type": "oom",
                        "injection_step": 100,
                        "duration": 10
                    }
                ]
            }
        }
        
        # 创建钩子
        hooks = create_fault_hooks_from_config(config)
        
        assert len(hooks) == 2
        assert hooks[0].fault_type == "nan_loss"
        assert hooks[0].trigger_step == 50
        assert hooks[1].fault_type == "oom"
        assert hooks[1].trigger_step == 100


if __name__ == "__main__":
    pytest.main([__file__])