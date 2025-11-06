"""
数据集管理器
支持多种数据集和任务类型的加载和管理
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from PIL import Image
import io

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """支持的数据集类型枚举"""
    IMDB_SENTIMENT = "imdb_sentiment"           # IMDB电影评论情感分析
    GLUE_COLA = "glue_cola"                     # 语言可接受性语料库
    GLUE_SST2 = "glue_sst2"                     # 斯坦福情感树库
    AG_NEWS = "ag_news"                         # 新闻分类
    CIFAR10 = "cifar10"                         # CIFAR-10图像分类
    CIFAR100 = "cifar100"                       # CIFAR-100图像分类
    FASHION_MNIST = "fashion_mnist"             # 时尚MNIST
    MNIST = "mnist"                             # 手写数字识别
    WIKITEXT = "wikitext"                       # 维基文本语言建模
    SQUAD = "squad"                             # 问答数据集
    CUSTOM_TEXT = "custom_text"                 # 自定义文本数据集
    CUSTOM_IMAGE = "custom_image"               # 自定义图像数据集


class TaskType(Enum):
    """支持的任务类型枚举"""
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    LANGUAGE_MODELING = "language_modeling"
    QUESTION_ANSWERING = "question_answering"
    SEQUENCE_LABELING = "sequence_labeling"


class DatasetManager:
    """数据集管理器类"""
    
    # 数据集配置映射
    DATASET_CONFIGS = {
        DatasetType.IMDB_SENTIMENT: {
            "task_type": TaskType.TEXT_CLASSIFICATION,
            "huggingface_name": "imdb",
            "text_column": "text",
            "label_column": "label",
            "num_labels": 2,
            "description": "IMDB电影评论情感分析数据集"
        },
        DatasetType.GLUE_COLA: {
            "task_type": TaskType.TEXT_CLASSIFICATION,
            "huggingface_name": "glue",
            "huggingface_config": "cola",
            "text_column": "sentence",
            "label_column": "label",
            "num_labels": 2,
            "description": "语言可接受性语料库"
        },
        DatasetType.GLUE_SST2: {
            "task_type": TaskType.TEXT_CLASSIFICATION,
            "huggingface_name": "glue",
            "huggingface_config": "sst2",
            "text_column": "sentence",
            "label_column": "label",
            "num_labels": 2,
            "description": "斯坦福情感树库"
        },
        DatasetType.AG_NEWS: {
            "task_type": TaskType.TEXT_CLASSIFICATION,
            "huggingface_name": "ag_news",
            "text_column": "text",
            "label_column": "label",
            "num_labels": 4,
            "description": "AG新闻分类数据集"
        },
        DatasetType.CIFAR10: {
            "task_type": TaskType.IMAGE_CLASSIFICATION,
            "huggingface_name": "cifar10",
            "image_column": "img",
            "label_column": "label",
            "num_labels": 10,
            "description": "CIFAR-10图像分类数据集"
        },
        DatasetType.CIFAR100: {
            "task_type": TaskType.IMAGE_CLASSIFICATION,
            "huggingface_name": "cifar100",
            "image_column": "img",
            "label_column": "label",
            "num_labels": 100,
            "description": "CIFAR-100图像分类数据集"
        },
        DatasetType.FASHION_MNIST: {
            "task_type": TaskType.IMAGE_CLASSIFICATION,
            "huggingface_name": "fashion_mnist",
            "image_column": "image",
            "label_column": "label",
            "num_labels": 10,
            "description": "时尚MNIST数据集"
        },
        DatasetType.MNIST: {
            "task_type": TaskType.IMAGE_CLASSIFICATION,
            "huggingface_name": "mnist",
            "image_column": "image",
            "label_column": "label",
            "num_labels": 10,
            "description": "手写数字识别数据集"
        },
        DatasetType.WIKITEXT: {
            "task_type": TaskType.LANGUAGE_MODELING,
            "huggingface_name": "wikitext",
            "huggingface_config": "wikitext-103-raw-v1",
            "text_column": "text",
            "description": "维基文本语言建模数据集"
        },
        DatasetType.SQUAD: {
            "task_type": TaskType.QUESTION_ANSWERING,
            "huggingface_name": "squad",
            "context_column": "context",
            "question_column": "question",
            "answers_column": "answers",
            "description": "SQuAD问答数据集"
        }
    }
    
    def __init__(self):
        self.loaded_datasets = {}
        self.dataset_statistics = {}
    
    def get_available_datasets(self) -> Dict[DatasetType, Dict[str, Any]]:
        """获取所有可用的数据集"""
        return self.DATASET_CONFIGS.copy()
    
    def load_dataset(self, dataset_type: DatasetType, split_sizes: Dict[str, int] = None,
                   cache_dir: str = None, **kwargs) -> DatasetDict:
        """
        加载数据集
        
        Args:
            dataset_type: 数据集类型
            split_sizes: 分割大小（用于限制数据集大小）
            cache_dir: 缓存目录
            **kwargs: 额外参数
            
        Returns:
            DatasetDict对象
        """
        logger.info(f"加载数据集: {dataset_type.value}")
        
        if dataset_type not in self.DATASET_CONFIGS:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        config = self.DATASET_CONFIGS[dataset_type]
        
        try:
            # 加载数据集
            if "huggingface_config" in config:
                dataset = load_dataset(
                    config["huggingface_name"],
                    config["huggingface_config"],
                    cache_dir=cache_dir,
                    **kwargs
                )
            else:
                dataset = load_dataset(
                    config["huggingface_name"],
                    cache_dir=cache_dir,
                    **kwargs
                )
            
            # 限制数据集大小（用于快速测试）
            if split_sizes:
                limited_dataset = {}
                for split_name, split_data in dataset.items():
                    if split_name in split_sizes:
                        max_size = split_sizes[split_name]
                        if len(split_data) > max_size:
                            limited_dataset[split_name] = split_data.select(range(max_size))
                            logger.info(f"限制{split_name}分割大小为: {max_size}")
                        else:
                            limited_dataset[split_name] = split_data
                    else:
                        limited_dataset[split_name] = split_data
                dataset = DatasetDict(limited_dataset)
            
            # 存储数据集统计信息
            self._compute_dataset_statistics(dataset, dataset_type)
            
            # 缓存数据集
            self.loaded_datasets[dataset_type] = dataset
            
            logger.info(f"数据集加载成功: {dataset_type.value}")
            return dataset
            
        except Exception as e:
            logger.error(f"加载数据集失败: {dataset_type.value}, 错误: {e}")
            raise
    
    def _compute_dataset_statistics(self, dataset: DatasetDict, dataset_type: DatasetType):
        """计算数据集统计信息"""
        stats = {
            "dataset_type": dataset_type.value,
            "splits": {},
            "total_samples": 0
        }
        
        config = self.DATASET_CONFIGS[dataset_type]
        
        for split_name, split_data in dataset.items():
            split_stats = {
                "samples": len(split_data),
                "features": list(split_data.features.keys())
            }
            
            # 标签分布（对于分类任务）
            if config["task_type"] in [TaskType.TEXT_CLASSIFICATION, TaskType.IMAGE_CLASSIFICATION]:
                label_column = config.get("label_column")
                if label_column and label_column in split_data.features:
                    labels = split_data[label_column]
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    split_stats["label_distribution"] = {
                        str(label): int(count) for label, count in zip(unique_labels, counts)
                    }
            
            stats["splits"][split_name] = split_stats
            stats["total_samples"] += len(split_data)
        
        self.dataset_statistics[dataset_type] = stats
        logger.info(f"数据集统计: {stats}")
    
    def get_dataset_info(self, dataset_type: DatasetType) -> Dict[str, Any]:
        """获取数据集信息"""
        if dataset_type not in self.DATASET_CONFIGS:
            raise ValueError(f"未知的数据集类型: {dataset_type}")
        
        config = self.DATASET_CONFIGS[dataset_type].copy()
        
        # 添加统计信息（如果已加载）
        if dataset_type in self.dataset_statistics:
            config["statistics"] = self.dataset_statistics[dataset_type]
        
        return config
    
    def create_custom_dataset(self, dataset_type: DatasetType, data: Dict[str, Any],
                            **kwargs) -> DatasetDict:
        """
        创建自定义数据集
        
        Args:
            dataset_type: 数据集类型（CUSTOM_TEXT或CUSTOM_IMAGE）
            data: 数据字典，键为分割名称，值为数据列表
            **kwargs: 额外参数
            
        Returns:
            DatasetDict对象
        """
        if dataset_type not in [DatasetType.CUSTOM_TEXT, DatasetType.CUSTOM_IMAGE]:
            raise ValueError("自定义数据集类型必须是CUSTOM_TEXT或CUSTOM_IMAGE")
        
        logger.info(f"创建自定义数据集: {dataset_type.value}")
        
        dataset_dict = {}
        
        for split_name, split_data in data.items():
            if dataset_type == DatasetType.CUSTOM_TEXT:
                # 文本数据
                if not isinstance(split_data, list) or not all(isinstance(item, (str, dict)) for item in split_data):
                    raise ValueError("自定义文本数据应该是字符串或字典列表")
                
                dataset_dict[split_name] = Dataset.from_list(split_data)
                
            elif dataset_type == DatasetType.CUSTOM_IMAGE:
                # 图像数据
                if not isinstance(split_data, list) or not all(isinstance(item, dict) for item in split_data):
                    raise ValueError("自定义图像数据应该是字典列表，包含'image'和'label'键")
                
                # 验证图像数据格式
                for item in split_data:
                    if 'image' not in item or 'label' not in item:
                        raise ValueError("图像数据必须包含'image'和'label'键")
                
                dataset_dict[split_name] = Dataset.from_list(split_data)
        
        dataset = DatasetDict(dataset_dict)
        
        # 存储自定义数据集配置
        custom_config = {
            "task_type": TaskType.TEXT_CLASSIFICATION if dataset_type == DatasetType.CUSTOM_TEXT else TaskType.IMAGE_CLASSIFICATION,
            "description": f"自定义{dataset_type.value}数据集",
            "custom": True
        }
        
        if dataset_type not in self.DATASET_CONFIGS:
            self.DATASET_CONFIGS[dataset_type] = custom_config
        
        self._compute_dataset_statistics(dataset, dataset_type)
        self.loaded_datasets[dataset_type] = dataset
        
        return dataset
    
    def preprocess_dataset(self, dataset: DatasetDict, dataset_type: DatasetType,
                         tokenizer=None, image_processor=None, **kwargs) -> DatasetDict:
        """
        预处理数据集
        
        Args:
            dataset: 数据集
            dataset_type: 数据集类型
            tokenizer: 文本tokenizer
            image_processor: 图像处理器
            **kwargs: 额外参数
            
        Returns:
            预处理后的数据集
        """
        logger.info(f"预处理数据集: {dataset_type.value}")
        
        config = self.DATASET_CONFIGS[dataset_type]
        task_type = config["task_type"]
        
        def text_preprocessing_function(examples):
            """文本预处理函数"""
            if tokenizer is None:
                return examples
            
            if task_type == TaskType.TEXT_CLASSIFICATION:
                return tokenizer(
                    examples[config["text_column"]],
                    truncation=True,
                    padding=False,
                    max_length=kwargs.get("max_length", 512)
                )
            elif task_type == TaskType.LANGUAGE_MODELING:
                return tokenizer(
                    examples[config["text_column"]],
                    truncation=True,
                    padding=False,
                    max_length=kwargs.get("max_length", 1024)
                )
            elif task_type == TaskType.QUESTION_ANSWERING:
                # 问答任务的特殊处理
                questions = examples[config["question_column"]]
                contexts = examples[config["context_column"]]
                
                return tokenizer(
                    questions,
                    contexts,
                    truncation="only_second",
                    max_length=kwargs.get("max_length", 384),
                    stride=kwargs.get("stride", 128),
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length"
                )
            return examples
        
        def image_preprocessing_function(examples):
            """图像预处理函数"""
            if image_processor is None:
                return examples
            
            images = examples[config["image_column"]]
            processed_images = []
            
            for image in images:
                if isinstance(image, Image.Image):
                    processed_image = image
                elif isinstance(image, (np.ndarray, torch.Tensor)):
                    processed_image = Image.fromarray(image.numpy() if hasattr(image, 'numpy') else image)
                else:
                    # 假设是PIL图像或可转换为PIL的图像
                    processed_image = Image.open(io.BytesIO(image)) if isinstance(image, bytes) else image
                
                processed_images.append(processed_image)
            
            return image_processor(processed_images, return_tensors="pt")
        
        # 应用预处理
        processed_dataset = {}
        
        for split_name, split_data in dataset.items():
            if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.LANGUAGE_MODELING, TaskType.QUESTION_ANSWERING]:
                processed_dataset[split_name] = split_data.map(
                    text_preprocessing_function,
                    batched=True,
                    remove_columns=split_data.column_names if kwargs.get("remove_original_columns", True) else None
                )
            elif task_type == TaskType.IMAGE_CLASSIFICATION:
                processed_dataset[split_name] = split_data.map(
                    image_preprocessing_function,
                    batched=True,
                    remove_columns=split_data.column_names if kwargs.get("remove_original_columns", True) else None
                )
            else:
                processed_dataset[split_name] = split_data
        
        return DatasetDict(processed_dataset)
    
    def get_data_loader(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True,
                      num_workers: int = 0, **kwargs) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            **kwargs: 额外参数
            
        Returns:
            DataLoader对象
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    
    def save_dataset(self, dataset: DatasetDict, save_path: str, format: str = "arrow"):
        """
        保存数据集
        
        Args:
            dataset: 数据集
            save_path: 保存路径
            format: 保存格式（arrow, csv, json）
        """
        os.makedirs(save_path, exist_ok=True)
        
        for split_name, split_data in dataset.items():
            split_path = os.path.join(save_path, f"{split_name}.{format}")
            
            if format == "arrow":
                split_data.save_to_disk(split_path)
            elif format == "csv":
                split_data.to_csv(split_path)
            elif format == "json":
                split_data.to_json(split_path)
            else:
                raise ValueError(f"不支持的保存格式: {format}")
        
        logger.info(f"数据集已保存到: {save_path}")


# 便捷函数
def get_all_available_datasets() -> Dict[str, Dict[str, Any]]:
    """获取所有可用数据集的简化接口"""
    manager = DatasetManager()
    available = manager.get_available_datasets()
    return {dataset_type.value: config for dataset_type, config in available.items()}


def load_dataset_by_name(dataset_name: str, **kwargs) -> DatasetDict:
    """通过名称加载数据集的简化接口"""
    manager = DatasetManager()
    
    try:
        dataset_type = DatasetType(dataset_name)
        return manager.load_dataset(dataset_type, **kwargs)
    except ValueError:
        raise ValueError(f"未知的数据集名称: {dataset_name}")


if __name__ == "__main__":
    # 测试代码
    manager = DatasetManager()
    
    # 获取可用数据集
    available_datasets = manager.get_available_datasets()
    print("可用数据集:")
    for dataset_type, config in available_datasets.items():
        print(f"  {dataset_type.value}: {config['description']}")
    
    # 测试加载数据集
    try:
        dataset = manager.load_dataset(
            DatasetType.IMDB_SENTIMENT,
            split_sizes={"train": 1000, "test": 200}
        )
        print(f"成功加载 IMDB 数据集:")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 样本")
        
        # 获取数据集信息
        info = manager.get_dataset_info(DatasetType.IMDB_SENTIMENT)
        print(f"数据集信息: {info}")
        
    except Exception as e:
        print(f"加载数据集失败: {e}")