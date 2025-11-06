"""
训练过程钩子机制
支持在训练过程中动态注入故障，包括NaN Loss、OOM、学习率异常等
"""

import torch
import numpy as np
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
import logging

logger = logging.getLogger(__name__)


class FaultConfig:
    """故障配置类"""
    
    def __init__(self, fault_type: str, trigger_step: int, **params):
        self.fault_type = fault_type
        self.trigger_step = trigger_step
        self.params = params
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建配置"""
        return cls(
            fault_type=config_dict['fault_type'],
            trigger_step=config_dict['trigger_step'],
            **config_dict.get('params', {})
        )


class FaultInjectionHook:
    """故障注入钩子基类"""
    
    def __init__(self, fault_type: str, trigger_step: int, **kwargs):
        self.fault_type = fault_type
        self.trigger_step = trigger_step
        self.params = kwargs
        self.activated = False
        self.original_state = {}
        
    def should_activate(self, current_step: int) -> bool:
        """判断是否应该激活故障"""
        return current_step >= self.trigger_step and not self.activated
    
    def activate(self, trainer, model, **kwargs):
        """激活故障注入"""
        raise NotImplementedError
    
    def deactivate(self, trainer, model, **kwargs):
        """停用故障注入，恢复原始状态"""
        pass


class NaNLossHook(FaultInjectionHook):
    """NaN Loss注入钩子"""
    
    def __init__(self, trigger_step: int, corruption_probability: float = 0.5, 
                 target_layers: List[str] = None, duration_steps: int = 5):
        super().__init__("nan_loss", trigger_step, 
                        corruption_probability=corruption_probability,
                        target_layers=target_layers or ["classifier"],
                        duration_steps=duration_steps)
        self.corruption_hooks = []
        self.step_count = 0
    
    def _corrupt_gradient(self, grad):
        """梯度损坏函数"""
        if torch.rand(1).item() < self.params['corruption_probability']:
            # 注入NaN或极大值
            corruption_type = torch.rand(1).item()
            if corruption_type < 0.5:
                return torch.full_like(grad, float('nan'))
            else:
                return torch.full_like(grad, 1e10)
        return grad
    
    def activate(self, trainer, model, **kwargs):
        """激活NaN Loss注入"""
        if self.activated:
            return
            
        logger.warning(f"[FAULT INJECTION] 激活NaN Loss注入 - Step {self.trigger_step}")
        
        # 为目标层注册梯度钩子
        for name, param in model.named_parameters():
            if any(target in name for target in self.params['target_layers']):
                hook = param.register_hook(self._corrupt_gradient)
                self.corruption_hooks.append(hook)
                logger.info(f"[FAULT INJECTION] 已为 {name} 注册梯度损坏钩子")
        
        self.activated = True
        self.step_count = 0
    
    def deactivate(self, trainer, model, **kwargs):
        """停用NaN Loss注入"""
        if not self.activated:
            return
            
        # 移除所有梯度钩子
        for hook in self.corruption_hooks:
            hook.remove()
        self.corruption_hooks.clear()
        
        logger.info(f"[FAULT INJECTION] 停用NaN Loss注入")
        self.activated = False
    
    def step(self):
        """步进计数，用于控制故障持续时间"""
        if self.activated:
            self.step_count += 1
            return self.step_count >= self.params['duration_steps']
        return False


class OOMHook(FaultInjectionHook):
    """OOM (内存溢出) 注入钩子"""
    
    def __init__(self, trigger_step: int, memory_size_mb: int = 1024, 
                 allocation_pattern: str = "exponential"):
        super().__init__("oom", trigger_step,
                        memory_size_mb=memory_size_mb,
                        allocation_pattern=allocation_pattern)
        self.allocated_tensors = []
    
    def activate(self, trainer, model, **kwargs):
        """激活OOM注入"""
        if self.activated:
            return
            
        logger.warning(f"[FAULT INJECTION] 激活OOM注入 - 分配 {self.params['memory_size_mb']}MB 内存")
        
        try:
            device = next(model.parameters()).device
            memory_size_bytes = self.params['memory_size_mb'] * 1024 * 1024
            
            if self.params['allocation_pattern'] == "exponential":
                # 指数增长分配
                current_size = 1024 * 1024  # 1MB起始
                while current_size < memory_size_bytes:
                    tensor_size = min(current_size, memory_size_bytes - sum(t.numel() * t.element_size() for t in self.allocated_tensors))
                    if tensor_size <= 0:
                        break
                    
                    tensor = torch.randn(tensor_size // 4, device=device, dtype=torch.float32)
                    self.allocated_tensors.append(tensor)
                    current_size *= 2
                    
            else:  # linear allocation
                # 线性分配
                chunk_size = 64 * 1024 * 1024  # 64MB chunks
                chunks_needed = memory_size_bytes // chunk_size
                
                for i in range(chunks_needed):
                    tensor = torch.randn(chunk_size // 4, device=device, dtype=torch.float32)
                    self.allocated_tensors.append(tensor)
            
            logger.info(f"[FAULT INJECTION] 已分配 {len(self.allocated_tensors)} 个内存块")
            self.activated = True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"[FAULT INJECTION] OOM注入成功触发: {e}")
                self.activated = True
            else:
                logger.error(f"[FAULT INJECTION] OOM注入失败: {e}")
    
    def deactivate(self, trainer, model, **kwargs):
        """停用OOM注入，释放内存"""
        if not self.activated:
            return
            
        # 释放所有分配的张量
        for tensor in self.allocated_tensors:
            del tensor
        self.allocated_tensors.clear()
        
        # 强制垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"[FAULT INJECTION] 停用OOM注入，已释放内存")
        self.activated = False


class NonConvergenceHook(FaultInjectionHook):
    """不收敛注入钩子"""
    
    def __init__(self, trigger_step: int, lr_multiplier: float = 1000.0, 
                 corruption_type: str = "too_high"):
        super().__init__("non_convergence", trigger_step,
                        lr_multiplier=lr_multiplier,
                        corruption_type=corruption_type)
    
    def activate(self, trainer, model, **kwargs):
        """激活不收敛注入"""
        if self.activated:
            return
            
        logger.warning(f"[FAULT INJECTION] 激活不收敛注入 - 学习率倍数: {self.params['lr_multiplier']}")
        
        # 保存原始学习率
        self.original_state['lr'] = trainer.optimizer.param_groups[0]['lr']
        
        # 修改学习率
        if self.params['corruption_type'] == "too_high":
            new_lr = self.original_state['lr'] * self.params['lr_multiplier']
        else:  # too_low
            new_lr = self.original_state['lr'] / self.params['lr_multiplier']
        
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        logger.info(f"[FAULT INJECTION] 学习率已从 {self.original_state['lr']} 修改为 {new_lr}")
        self.activated = True
    
    def deactivate(self, trainer, model, **kwargs):
        """停用不收敛注入，恢复学习率"""
        if not self.activated or 'lr' not in self.original_state:
            return
            
        # 恢复原始学习率
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = self.original_state['lr']
        
        logger.info(f"[FAULT INJECTION] 学习率已恢复为 {self.original_state['lr']}")
        self.activated = False


class GradientExplosionHook(FaultInjectionHook):
    """梯度爆炸注入钩子"""
    
    def __init__(self, trigger_step: int, explosion_factor: float = 1000.0,
                 target_layers: List[str] = None, duration_steps: int = 3):
        super().__init__("gradient_explosion", trigger_step,
                        explosion_factor=explosion_factor,
                        target_layers=target_layers or ["classifier"],
                        duration_steps=duration_steps)
        self.corruption_hooks = []
        self.step_count = 0
    
    def _explode_gradient(self, grad):
        """梯度爆炸函数"""
        if torch.rand(1).item() < 0.7:  # 70%概率触发梯度爆炸
            return grad * self.params['explosion_factor']
        return grad
    
    def activate(self, trainer, model, **kwargs):
        """激活梯度爆炸注入"""
        if self.activated:
            return
            
        logger.warning(f"[FAULT INJECTION] 激活梯度爆炸注入 - Step {self.trigger_step}, 倍数: {self.params['explosion_factor']}")
        
        # 为目标层注册梯度钩子
        for name, param in model.named_parameters():
            if any(target in name for target in self.params['target_layers']):
                hook = param.register_hook(self._explode_gradient)
                self.corruption_hooks.append(hook)
                logger.info(f"[FAULT INJECTION] 已为 {name} 注册梯度爆炸钩子")
        
        self.activated = True
        self.step_count = 0
    
    def deactivate(self, trainer, model, **kwargs):
        """停用梯度爆炸注入"""
        if not self.activated:
            return
            
        # 移除所有梯度钩子
        for hook in self.corruption_hooks:
            hook.remove()
        self.corruption_hooks.clear()
        
        logger.info(f"[FAULT INJECTION] 停用梯度爆炸注入")
        self.activated = False
    
    def step(self):
        """步进计数，用于控制故障持续时间"""
        if self.activated:
            self.step_count += 1
            return self.step_count >= self.params['duration_steps']
        return False


class TrainingFaultInjector(TrainerCallback):
    """训练故障注入器 - 集成到Transformers Trainer"""
    
    def __init__(self):
        self.hooks: List[FaultInjectionHook] = []
        self.active_hooks: List[FaultInjectionHook] = []
        
    def add_hook(self, hook: FaultInjectionHook):
        """添加故障注入钩子"""
        self.hooks.append(hook)
        logger.info(f"已添加故障注入钩子: {hook.fault_type} @ step {hook.trigger_step}")
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, **kwargs):
        """训练步骤开始时的回调"""
        current_step = state.global_step
        
        # 检查是否需要激活新的钩子
        for hook in self.hooks:
            if hook.should_activate(current_step):
                hook.activate(kwargs.get('trainer'), kwargs.get('model'))
                self.active_hooks.append(hook)
        
        # 检查是否需要停用钩子
        hooks_to_remove = []
        for hook in self.active_hooks:
            if hasattr(hook, 'step') and hook.step():
                hook.deactivate(kwargs.get('trainer'), kwargs.get('model'))
                hooks_to_remove.append(hook)
        
        for hook in hooks_to_remove:
            self.active_hooks.remove(hook)
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """训练结束时清理所有钩子"""
        for hook in self.active_hooks:
            hook.deactivate(kwargs.get('trainer'), kwargs.get('model'))
        self.active_hooks.clear()


def create_fault_hooks_from_config(config: Dict[str, Any]) -> List[FaultInjectionHook]:
    """从配置创建故障注入钩子"""
    hooks = []
    
    # 基于步数的注入
    step_based = config.get("injection_schedule", {}).get("step_based", [])
    for schedule in step_based:
        fault_type = schedule.get("fault_type")
        injection_step = schedule.get("injection_step", 0)
        duration = schedule.get("duration", 5)
        
        if fault_type == "nan_loss":
            hook = NaNLossHook(
                trigger_step=injection_step,
                corruption_probability=0.5,
                duration_steps=duration
            )
            hooks.append(hook)
        elif fault_type == "oom":
            hook = OOMHook(
                trigger_step=injection_step,
                memory_size_mb=1024
            )
            hooks.append(hook)
        elif fault_type == "non_convergence":
            hook = NonConvergenceHook(
                trigger_step=injection_step,
                lr_multiplier=1000.0
            )
            hooks.append(hook)
    
    return hooks