"""
模型工厂
支持多种开源机器学习模型的创建和管理
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoTokenizer,
    ViTForImageClassification,
    SwinForImageClassification,
    ResNetForImageClassification,
    BertTokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel
)
from torchvision import models
import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """支持的模型类型枚举"""
    BERT_CLASSIFICATION = "bert_classification"
    VISION_TRANSFORMER = "vision_transformer"
    SWIN_TRANSFORMER = "swin_transformer"
    RESNET = "resnet"
    GPT2_LANGUAGE = "gpt2_language"
    EFFICIENTNET = "efficientnet"
    MOBILENET = "mobilenet"
    DISTILBERT = "distilbert"
    ROBERTA = "roberta"


class ModelFactory:
    """模型工厂类"""
    
    # 预训练模型映射
    MODEL_MAPPING = {
        ModelType.BERT_CLASSIFICATION: {
            "models": [
                "bert-base-uncased",
                "bert-large-uncased",
                "bert-base-cased",
                "bert-base-multilingual-cased"
            ],
            "tokenizer": BertTokenizer,
            "model_class": AutoModelForSequenceClassification
        },
        ModelType.DISTILBERT: {
            "models": [
                "distilbert-base-uncased",
                "distilbert-base-cased"
            ],
            "tokenizer": AutoTokenizer,
            "model_class": AutoModelForSequenceClassification
        },
        ModelType.ROBERTA: {
            "models": [
                "roberta-base",
                "roberta-large"
            ],
            "tokenizer": AutoTokenizer,
            "model_class": AutoModelForSequenceClassification
        },
        ModelType.VISION_TRANSFORMER: {
            "models": [
                "google/vit-base-patch16-224",
                "google/vit-large-patch16-224",
                "google/vit-base-patch16-224-in21k"
            ],
            "model_class": ViTForImageClassification
        },
        ModelType.SWIN_TRANSFORMER: {
            "models": [
                "microsoft/swin-base-patch4-window7-224",
                "microsoft/swin-small-patch4-window7-224"
            ],
            "model_class": SwinForImageClassification
        },
        ModelType.RESNET: {
            "models": [
                "microsoft/resnet-50",
                "microsoft/resnet-101",
                "microsoft/resnet-152"
            ],
            "model_class": ResNetForImageClassification
        },
        ModelType.GPT2_LANGUAGE: {
            "models": [
                "gpt2",
                "gpt2-medium",
                "gpt2-large",
                "gpt2-xl"
            ],
            "tokenizer": GPT2Tokenizer,
            "model_class": GPT2LMHeadModel
        }
    }
    
    # 计算机视觉模型（使用torchvision）
    TORCHVISION_MODELS = {
        ModelType.EFFICIENTNET: {
            "models": {
                "efficientnet_b0": models.efficientnet_b0,
                "efficientnet_b1": models.efficientnet_b1,
                "efficientnet_b2": models.efficientnet_b2,
                "efficientnet_b3": models.efficientnet_b3,
                "efficientnet_b4": models.efficientnet_b4,
            }
        },
        ModelType.MOBILENET: {
            "models": {
                "mobilenet_v2": models.mobilenet_v2,
                "mobilenet_v3_small": models.mobilenet_v3_small,
                "mobilenet_v3_large": models.mobilenet_v3_large,
            }
        }
    }
    
    @classmethod
    def get_available_models(cls) -> Dict[ModelType, List[str]]:
        """获取所有可用的模型"""
        available_models = {}
        
        # Hugging Face 模型
        for model_type, config in cls.MODEL_MAPPING.items():
            available_models[model_type] = config["models"]
        
        # Torchvision 模型
        for model_type, config in cls.TORCHVISION_MODELS.items():
            available_models[model_type] = list(config["models"].keys())
        
        return available_models
    
    @classmethod
    def create_model(cls, model_type: ModelType, model_name: str, num_labels: int = None,
                   **kwargs) -> tuple:
        """
        创建模型和对应的tokenizer/preprocessor
        
        Args:
            model_type: 模型类型
            model_name: 模型名称
            num_labels: 分类任务的标签数量
            **kwargs: 额外参数
            
        Returns:
            (model, tokenizer/preprocessor)
        """
        logger.info(f"创建模型: {model_type.value} - {model_name}")
        
        try:
            # Hugging Face 模型
            if model_type in cls.MODEL_MAPPING:
                config = cls.MODEL_MAPPING[model_type]
                
                # 创建tokenizer（如果适用）
                tokenizer = None
                if "tokenizer" in config:
                    tokenizer_class = config["tokenizer"]
                    tokenizer = tokenizer_class.from_pretrained(model_name)
                
                # 创建模型
                model_class = config["model_class"]
                
                if model_type in [ModelType.BERT_CLASSIFICATION, ModelType.DISTILBERT, ModelType.ROBERTA]:
                    if num_labels is None:
                        raise ValueError("分类任务需要指定num_labels参数")
                    
                    model = model_class.from_pretrained(
                        model_name,
                        num_labels=num_labels,
                        **kwargs
                    )
                elif model_type == ModelType.GPT2_LANGUAGE:
                    model = model_class.from_pretrained(model_name, **kwargs)
                else:  # 视觉模型
                    model = model_class.from_pretrained(model_name, **kwargs)
                
                return model, tokenizer
            
            # Torchvision 模型
            elif model_type in cls.TORCHVISION_MODELS:
                config = cls.TORCHVISION_MODELS[model_type]
                
                if model_name not in config["models"]:
                    raise ValueError(f"未知的torchvision模型: {model_name}")
                
                model_creator = config["models"][model_name]
                model = model_creator(pretrained=True, **kwargs)
                
                # 修改分类头（如果需要）
                if num_labels is not None:
                    if hasattr(model, 'classifier'):
                        if isinstance(model.classifier, nn.Sequential):
                            # 对于EfficientNet等模型
                            in_features = model.classifier[-1].in_features
                            model.classifier[-1] = nn.Linear(in_features, num_labels)
                        else:
                            # 对于MobileNet等模型
                            in_features = model.classifier.in_features
                            model.classifier = nn.Linear(in_features, num_labels)
                    elif hasattr(model, 'fc'):
                        # 对于ResNet等模型
                        in_features = model.fc.in_features
                        model.fc = nn.Linear(in_features, num_labels)
                    elif hasattr(model, 'heads'):
                        # 对于Vision Transformer等模型
                        in_features = model.heads.head.in_features
                        model.heads.head = nn.Linear(in_features, num_labels)
                
                return model, None
            
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
        except Exception as e:
            logger.error(f"创建模型失败: {model_type.value} - {model_name}, 错误: {e}")
            raise
    
    @classmethod
    def get_model_info(cls, model_type: ModelType, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "model_type": model_type.value,
            "model_name": model_name,
            "parameters": "未知",
            "architecture": "未知"
        }
        
        try:
            if model_type in cls.MODEL_MAPPING:
                # Hugging Face 模型
                config = cls.MODEL_MAPPING[model_type]
                if model_name in config["models"]:
                    model_class = config["model_class"]
                    model = model_class.from_pretrained(model_name)
                    info["parameters"] = sum(p.numel() for p in model.parameters())
                    info["architecture"] = model.__class__.__name__
                    del model
                    
            elif model_type in cls.TORCHVISION_MODELS:
                # Torchvision 模型
                config = cls.TORCHVISION_MODELS[model_type]
                if model_name in config["models"]:
                    model_creator = config["models"][model_name]
                    model = model_creator(pretrained=True)
                    info["parameters"] = sum(p.numel() for p in model.parameters())
                    info["architecture"] = model.__class__.__name__
                    del model
        
        except Exception as e:
            logger.warning(f"获取模型信息失败: {e}")
        
        return info
    
    @classmethod
    def validate_model_config(cls, model_config: Dict[str, Any]) -> bool:
        """验证模型配置"""
        required_keys = ["model_type", "model_name"]
        
        for key in required_keys:
            if key not in model_config:
                logger.error(f"模型配置缺少必需键: {key}")
                return False
        
        try:
            model_type = ModelType(model_config["model_type"])
            model_name = model_config["model_name"]
            
            # 检查模型是否可用
            available_models = cls.get_available_models()
            if model_type not in available_models:
                logger.error(f"不支持的模型类型: {model_type}")
                return False
            
            if model_name not in available_models[model_type]:
                logger.error(f"模型名称不可用: {model_name}")
                return False
            
            return True
            
        except ValueError as e:
            logger.error(f"模型配置验证失败: {e}")
            return False


class MultiModelTrainer:
    """多模型训练器"""
    
    def __init__(self, model_factory: ModelFactory = None):
        self.model_factory = model_factory or ModelFactory()
        self.trained_models = {}
        self.training_logs = {}
    
    def train_multiple_models(self, model_configs: List[Dict[str, Any]], 
                            dataset_config: Dict[str, Any],
                            training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        训练多个模型
        
        Args:
            model_configs: 模型配置列表
            dataset_config: 数据集配置
            training_config: 训练配置
            
        Returns:
            训练结果汇总
        """
        logger.info(f"开始训练 {len(model_configs)} 个模型")
        
        results = {
            "successful_models": [],
            "failed_models": [],
            "total_training_time": 0,
            "model_performance": {}
        }
        
        for i, model_config in enumerate(model_configs):
            logger.info(f"训练模型 {i+1}/{len(model_configs)}: {model_config}")
            
            try:
                # 验证模型配置
                if not self.model_factory.validate_model_config(model_config):
                    logger.warning(f"模型配置无效: {model_config}")
                    results["failed_models"].append({
                        "config": model_config,
                        "error": "配置验证失败"
                    })
                    continue
                
                # 训练单个模型
                model_result = self._train_single_model(
                    model_config, dataset_config, training_config
                )
                
                if model_result["success"]:
                    results["successful_models"].append(model_config)
                    results["model_performance"][model_config["model_name"]] = model_result
                    results["total_training_time"] += model_result.get("training_time", 0)
                else:
                    results["failed_models"].append({
                        "config": model_config,
                        "error": model_result.get("error", "未知错误")
                    })
                    
            except Exception as e:
                logger.error(f"训练模型失败: {model_config}, 错误: {e}")
                results["failed_models"].append({
                    "config": model_config,
                    "error": str(e)
                })
        
        logger.info(f"训练完成: {len(results['successful_models'])} 成功, {len(results['failed_models'])} 失败")
        return results
    
    def _train_single_model(self, model_config: Dict[str, Any],
                          dataset_config: Dict[str, Any],
                          training_config: Dict[str, Any]) -> Dict[str, Any]:
        """训练单个模型"""
        # 这里可以集成现有的训练逻辑
        # 暂时返回模拟结果
        import time
        import random
        
        start_time = time.time()
        
        # 模拟训练过程
        time.sleep(2)  # 模拟训练时间
        
        training_time = time.time() - start_time
        
        # 模拟性能指标
        performance = {
            "accuracy": random.uniform(0.7, 0.95),
            "loss": random.uniform(0.1, 0.5),
            "training_time": training_time,
            "memory_usage": random.uniform(100, 5000)  # MB
        }
        
        return {
            "success": True,
            "performance": performance,
            "training_time": training_time,
            "model_config": model_config
        }


# 便捷函数
def get_all_available_models() -> Dict[str, List[str]]:
    """获取所有可用模型的简化接口"""
    available = ModelFactory.get_available_models()
    return {model_type.value: models for model_type, models in available.items()}


def create_model_from_config(model_config: Dict[str, Any], **kwargs) -> tuple:
    """从配置创建模型的简化接口"""
    model_type = ModelType(model_config["model_type"])
    model_name = model_config["model_name"]
    num_labels = model_config.get("num_labels")
    
    return ModelFactory.create_model(model_type, model_name, num_labels, **kwargs)


if __name__ == "__main__":
    # 测试代码
    factory = ModelFactory()
    
    # 获取可用模型
    available_models = factory.get_available_models()
    print("可用模型:")
    for model_type, models in available_models.items():
        print(f"  {model_type.value}: {models}")
    
    # 测试创建模型
    try:
        model, tokenizer = factory.create_model(
            ModelType.BERT_CLASSIFICATION,
            "bert-base-uncased",
            num_labels=2
        )
        print(f"成功创建 BERT 模型: {type(model)}")
        
        # 获取模型信息
        info = factory.get_model_info(ModelType.BERT_CLASSIFICATION, "bert-base-uncased")
        print(f"模型信息: {info}")
        
    except Exception as e:
        print(f"创建模型失败: {e}")