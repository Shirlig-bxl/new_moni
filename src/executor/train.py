"""
主训练脚本
基于Hugging Face Transformers的BERT-IMDB微调
支持故障注入和详细监控日志
支持GPU密集型训练模式
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.trainer_callback import TrainerCallback
import evaluate

from .config import TrainingConfig, FaultConfigFactory
from src.injector.training_hook import TrainingFaultInjector, create_fault_hooks_from_config


# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DetailedLoggingCallback(TrainerCallback):
    """详细日志回调，记录训练过程中的关键指标"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info("=" * 50)
        logger.info("训练开始")
        logger.info(f"总步数: {state.max_steps}")
        logger.info(f"总轮数: {args.num_train_epochs}")
        logger.info(f"批次大小: {args.per_device_train_batch_size}")
        logger.info(f"学习率: {args.learning_rate}")
        logger.info("=" * 50)
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_time = time.time()
            if hasattr(self, 'step_start_time'):
                step_duration = current_time - self.step_start_time
                self.step_times.append(step_duration)
                
                # 计算吞吐量 (samples/second)
                throughput = args.per_device_train_batch_size / step_duration if step_duration > 0 else 0
                
                # 记录详细指标
                log_msg = f"Step {state.global_step}"
                if 'loss' in logs:
                    log_msg += f" | Loss: {logs['loss']:.6f}"
                if 'learning_rate' in logs:
                    log_msg += f" | LR: {logs['learning_rate']:.2e}"
                log_msg += f" | Throughput: {throughput:.2f} samples/s"
                log_msg += f" | Step Time: {step_duration:.3f}s"
                
                logger.info(log_msg)
                
                # 检查异常值
                if 'loss' in logs:
                    if np.isnan(logs['loss']) or np.isinf(logs['loss']):
                        logger.error(f"[ANOMALY DETECTED] NaN/Inf Loss at step {state.global_step}: {logs['loss']}")
                    elif logs['loss'] > 100:
                        logger.warning(f"[ANOMALY DETECTED] High Loss at step {state.global_step}: {logs['loss']}")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info("=" * 30)
            logger.info("评估结果:")
            for key, value in logs.items():
                if key.startswith('eval_'):
                    logger.info(f"{key}: {value:.6f}")
            logger.info("=" * 30)
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        
        logger.info("=" * 50)
        logger.info("训练结束")
        logger.info(f"总训练时间: {total_time:.2f}s")
        logger.info(f"平均步时间: {avg_step_time:.3f}s")
        logger.info(f"总步数: {len(self.step_times)}")
        logger.info("=" * 50)


def compute_metrics(eval_pred):
    """计算评估指标"""
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def preprocess_function(examples, tokenizer, max_length=512):
    """数据预处理函数"""
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding=False,  # 使用DataCollator进行动态padding
        max_length=max_length
    )


def load_and_prepare_data(config: TrainingConfig):
    """加载和准备IMDB数据集"""
    logger.info("加载IMDB数据集...")
    
    try:
        # 加载数据集
        dataset = load_dataset("imdb")
        
        # 为了快速测试，可以使用数据集的子集
        # 在实际实验中可以使用完整数据集
        train_dataset = dataset["train"].select(range(5000))  # 使用5000个训练样本
        eval_dataset = dataset["test"].select(range(1000))    # 使用1000个测试样本
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"测试集大小: {len(eval_dataset)}")
        
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        logger.info("尝试创建模拟数据集...")
        # 创建模拟数据集作为回退
        from datasets import Dataset
        import pandas as pd
        
        # 创建小型模拟数据集
        train_data = {
            "text": ["This is a great movie!", "Terrible film, waste of time."] * 100,
            "label": [1, 0] * 100
        }
        eval_data = {
            "text": ["Amazing performance!", "Boring and predictable."] * 50,
            "label": [1, 0] * 50
        }
        
        train_dataset = Dataset.from_dict(train_data)
        eval_dataset = Dataset.from_dict(eval_data)
        logger.warning(f"使用模拟数据集: 训练集 {len(train_dataset)}, 测试集 {len(eval_dataset)}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except Exception as e:
        logger.error(f"Tokenizer加载失败: {e}")
        logger.info("尝试使用默认tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 预处理数据
    logger.info("预处理数据...")
    try:
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
            batched=True
        )
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
            batched=True
        )
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        raise
    
    return train_dataset, eval_dataset, tokenizer


def create_model(config: TrainingConfig):
    """创建模型"""
    logger.info(f"加载模型: {config.model_name}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,  # IMDB是二分类任务
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    return model


def setup_trainer(model, train_dataset, eval_dataset, tokenizer, config: TrainingConfig):
    """设置Trainer"""
    
    # 创建训练参数
    training_args = TrainingArguments(**config.get_training_args_dict())
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 创建回调
    callbacks = [DetailedLoggingCallback()]
    
    # 添加故障注入回调（如果配置中存在）
    if hasattr(config, 'fault_injection') and config.fault_injection:
        fault_injector = TrainingFaultInjector()
        hooks = create_fault_hooks_from_config(config.fault_injection)
        for hook in hooks:
            fault_injector.add_hook(hook)
        callbacks.append(fault_injector)
        logger.info(f"已添加故障注入回调，包含 {len(hooks)} 个故障钩子")
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    return trainer


def main(config_name: Optional[str] = None, config_path: Optional[str] = None):
    """主训练函数"""
    
    try:
        # 加载配置
        if config_path:
            config = TrainingConfig.from_yaml(config_path)
            logger.info(f"从文件加载配置: {config_path}")
        elif config_name:
            configs = FaultConfigFactory.get_all_fault_configs()
            if config_name not in configs:
                logger.error(f"未知配置名称: {config_name}. 可用配置: {list(configs.keys())}")
                raise ValueError(f"未知配置名称: {config_name}")
            config = configs[config_name]
            logger.info(f"使用预定义配置: {config_name}")
        else:
            config = FaultConfigFactory.create_baseline_config()
            logger.info("使用默认基线配置")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        logger.info("尝试使用最小配置继续...")
        # 创建最小配置
        config = TrainingConfig()
        config.model_name = "bert-base-uncased"
        config.output_dir = "./output"
        config.logging_dir = "./logs"
        config.max_seq_length = 512
        config.per_device_train_batch_size = 8
        config.per_device_eval_batch_size = 8
        config.num_train_epochs = 1  # 减少轮数以快速验证
        config.learning_rate = 2e-5
        config.weight_decay = 0.01
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # 保存当前配置
    config_save_path = os.path.join(config.output_dir, "training_config.yaml")
    config.to_yaml(config_save_path)
    logger.info(f"配置已保存到: {config_save_path}")
    
    # 记录系统信息
    logger.info("=" * 60)
    logger.info("系统信息:")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
        logger.info(f"当前CUDA设备: {torch.cuda.current_device()}")
        logger.info(f"GPU名称: {torch.cuda.get_device_name()}")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        train_dataset, eval_dataset, tokenizer = load_and_prepare_data(config)
        
        # 创建模型
        model = create_model(config)
        
        # 设置训练器
        trainer = setup_trainer(model, train_dataset, eval_dataset, tokenizer, config)
        
        # 开始训练
        logger.info("开始训练...")
        start_time = time.time()
        
        trainer.train()
        
        end_time = time.time()
        logger.info(f"训练完成，总耗时: {end_time - start_time:.2f}秒")
        
        # 最终评估
        logger.info("进行最终评估...")
        eval_results = trainer.evaluate()
        logger.info("最终评估结果:")
        for key, value in eval_results.items():
            logger.info(f"{key}: {value}")
        
        # 保存模型
        logger.info("保存模型...")
        trainer.save_model()
        
        return eval_results
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        
        # 记录CUDA OOM错误
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.error("[ANOMALY DETECTED] CUDA Out of Memory Error!")
        
        raise e


class GPUTrainingMode:
    """GPU密集型训练模式"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_with_timestamp(self, message):
        """带时间戳的日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info(f"[{timestamp}] {message}")
    
    def run_vision_transformer_training(self, device):
        """运行Vision Transformer训练"""
        self.log_with_timestamp("🚀 开始Vision Transformer (ViT-Large) 训练")
        
        # 创建ViT-Large模型 (约300M参数)
        model = self.create_vision_transformer().to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        self.log_with_timestamp(f"ViT模型参数: {total_params:,} ({total_params/1e6:.1f}M)")
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        batch_size = 32  # ViT需要较大内存
        
        for epoch in range(3):
            self.log_with_timestamp(f"ViT Epoch {epoch+1}/3")
            
            for batch_idx in range(50):
                # 创建图像数据 (B, C, H, W)
                images = torch.randn(batch_size, 3, 224, 224, device=device)
                targets = torch.randint(0, 1000, (batch_size,), device=device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    self.log_with_timestamp(f"  ViT Batch {batch_idx+1}/50, Loss: {loss.item():.4f}, GPU: {gpu_memory:.0f}MB")
                
                # 额外的GPU密集计算
                if batch_idx % 3 == 0:
                    extra_attn = torch.matmul(
                        torch.randn(batch_size, 197, 1024, device=device),
                        torch.randn(batch_size, 1024, 197, device=device)
                    )
    
    def create_vision_transformer(self):
        """创建Vision Transformer模型"""
        class VisionTransformer(nn.Module):
            def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                         embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0):
                super().__init__()
                self.img_size = img_size
                self.patch_size = patch_size
                self.num_patches = (img_size // patch_size) ** 2
                self.embed_dim = embed_dim
                
                # Patch embedding
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                
                # Position embedding
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                
                # Transformer blocks
                self.blocks = nn.ModuleList([
                    self.TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
                ])
                
                # Classification head
                self.norm = nn.LayerNorm(embed_dim)
                self.head = nn.Linear(embed_dim, num_classes)
                
            class TransformerBlock(nn.Module):
                def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(embed_dim)
                    self.attn = self.MultiHeadAttention(embed_dim, num_heads)
                    self.norm2 = nn.LayerNorm(embed_dim)
                    
                    mlp_hidden_dim = int(embed_dim * mlp_ratio)
                    self.mlp = nn.Sequential(
                        nn.Linear(embed_dim, mlp_hidden_dim),
                        nn.GELU(),
                        nn.Linear(mlp_hidden_dim, embed_dim)
                    )
                
                def forward(self, x):
                    x = x + self.attn(self.norm1(x))
                    x = x + self.mlp(self.norm2(x))
                    return x
            
            class MultiHeadAttention(nn.Module):
                def __init__(self, embed_dim, num_heads):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.num_heads = num_heads
                    self.head_dim = embed_dim // num_heads
                    
                    self.qkv = nn.Linear(embed_dim, embed_dim * 3)
                    self.proj = nn.Linear(embed_dim, embed_dim)
                
                def forward(self, x):
                    B, N, C = x.shape
                    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
                    attn = attn.softmax(dim=-1)
                    
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    return x
            
            def forward(self, x):
                B = x.shape[0]
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
                
                # Add cls token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                
                # Add position embedding
                x = x + self.pos_embed
                
                # Transformer blocks
                for block in self.blocks:
                    x = block(x)
                
                # Classification
                x = self.norm(x)
                cls_token_final = x[:, 0]
                return self.head(cls_token_final)
        
        return VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16
        )
    
    def run_intensive_training(self):
        """运行GPU密集型训练"""
        self.log_with_timestamp("🔥 GPU密集型训练开始")
        
        if not torch.cuda.is_available():
            self.log_with_timestamp("ERROR: CUDA不可用，无法继续")
            return
        
        device = torch.device('cuda')
        self.log_with_timestamp(f"使用设备: {device}")
        self.log_with_timestamp(f"GPU名称: {torch.cuda.get_device_name(0)}")
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        try:
            # 运行Vision Transformer训练
            self.run_vision_transformer_training(device)
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.log_with_timestamp("🎯 成功触发GPU内存限制！这说明GPU被充分利用了")
                self.log_with_timestamp("💡 可以适当减小batch_size继续训练")
            else:
                self.log_with_timestamp(f"训练错误: {e}")
        
        finally:
            # 显示最终GPU状态
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                gpu_cached = torch.cuda.memory_reserved() / 1024**2
                self.log_with_timestamp(f"最终GPU内存 - 已分配: {gpu_memory:.0f}MB, 已缓存: {gpu_cached:.0f}MB")
            
            self.log_with_timestamp("🎉 GPU密集型训练完成！")


def main_extended(config_name: Optional[str] = None, config_path: Optional[str] = None,
                 training_mode: str = "bert"):
    """扩展的主训练函数，支持多种训练模式"""
    
    if training_mode == "gpu_intensive":
        # GPU密集型训练模式
        gpu_trainer = GPUTrainingMode()
        gpu_trainer.run_intensive_training()
        return
    
    # 原有的BERT训练模式
    main(config_name, config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT-IMDB训练脚本")
    parser.add_argument("--config-name", type=str, help="预定义配置名称")
    parser.add_argument("--config-path", type=str, help="配置文件路径")
    parser.add_argument("--training-mode", type=str, choices=["bert", "gpu_intensive"],
                       default="bert", help="训练模式: bert (默认) 或 gpu_intensive")
    
    args = parser.parse_args()
    
    main_extended(config_name=args.config_name, config_path=args.config_path,
                 training_mode=args.training_mode)