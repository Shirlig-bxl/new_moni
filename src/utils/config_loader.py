"""
配置文件加载工具
Configuration Loader Utility
"""

import yaml
import os
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_dir: str = None, max_retries: int = 3, retry_delay: float = 1.0):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录路径
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if config_dir is None:
            # 默认配置目录
            current_dir = Path(__file__).parent.parent.parent
            self.config_dir = current_dir / "configs"
        else:
            self.config_dir = Path(config_dir)
            
        if not self.config_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")
        
        logger.info(f"配置加载器初始化完成，配置目录: {self.config_dir}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        加载指定的配置文件
        
        Args:
            config_name: 配置文件名 (不包含扩展名)
            
        Returns:
            配置字典
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def load_training_config(self) -> Dict[str, Any]:
        """加载训练配置"""
        return self.load_config("training_config")
    
    def load_fault_injection_config(self) -> Dict[str, Any]:
        """加载故障注入配置"""
        return self.load_config("fault_injection_config")
    
    def load_monitoring_config(self) -> Dict[str, Any]:
        """加载监控配置"""
        return self.load_config("monitoring_config")
    
    def get_config_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        获取嵌套配置值
        
        Args:
            config: 配置字典
            key_path: 键路径，用点分隔 (例如: "model.name")
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置文件
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def validate_config(self, config: Dict[str, Any], required_keys: list) -> bool:
        """
        验证配置文件是否包含必需的键
        
        Args:
            config: 配置字典
            required_keys: 必需的键列表
            
        Returns:
            是否有效
        """
        missing_keys = []
        for key_path in required_keys:
            if self.get_config_value(config, key_path) is None:
                missing_keys.append(key_path)
        
        if missing_keys:
            logger.error(f"配置验证失败，缺少必需的键: {missing_keys}")
            return False
        
        logger.info("配置验证成功")
        return True
    
    def load_config_with_fallback(self, config_name: str, fallback_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        加载配置文件，如果失败则使用回退配置
        
        Args:
            config_name: 配置文件名
            fallback_config: 回退配置
            
        Returns:
            配置字典
        """
        try:
            return self.load_config(config_name)
        except Exception as e:
            logger.warning(f"加载配置文件 {config_name} 失败，使用回退配置: {e}")
            if fallback_config is not None:
                return fallback_config
            else:
                # 提供默认回退配置
                default_fallbacks = {
                    "training_config": {
                        "model_name": "bert-base-uncased",
                        "output_dir": "./output",
                        "logging_dir": "./logs",
                        "max_seq_length": 512,
                        "per_device_train_batch_size": 8,
                        "per_device_eval_batch_size": 8,
                        "num_train_epochs": 3,
                        "learning_rate": 2e-5,
                        "weight_decay": 0.01
                    },
                    "fault_injection_config": {
                        "injection_schedule": {
                            "step_based": [],
                            "time_based": []
                        }
                    },
                    "monitoring_config": {
                        "gpu_monitoring": {"enabled": True},
                        "system_monitoring": {"enabled": True},
                        "log_monitoring": {"enabled": True}
                    }
                }
                return default_fallbacks.get(config_name, {})
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """
        保存配置文件
        
        Args:
            config: 配置字典
            config_name: 配置文件名 (不包含扩展名)
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            raise IOError(f"保存配置文件失败: {e}")


class ExperimentConfigManager:
    """实验配置管理器"""
    
    def __init__(self, config_loader: ConfigLoader = None):
        """
        初始化实验配置管理器
        
        Args:
            config_loader: 配置加载器实例
        """
        self.config_loader = config_loader or ConfigLoader()
    
    def create_experiment_config(self, 
                               experiment_name: str,
                               fault_type: Optional[str] = None,
                               custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建实验配置
        
        Args:
            experiment_name: 实验名称
            fault_type: 故障类型 (可选)
            custom_params: 自定义参数 (可选)
            
        Returns:
            实验配置字典
        """
        # 加载基础配置
        training_config = self.config_loader.load_training_config()
        fault_config = self.config_loader.load_fault_injection_config()
        monitoring_config = self.config_loader.load_monitoring_config()
        
        # 创建实验配置
        experiment_config = {
            "experiment_name": experiment_name,
            "training": training_config,
            "monitoring": monitoring_config,
            "fault_injection": {
                "enabled": fault_type is not None,
                "fault_type": fault_type,
                "config": fault_config
            }
        }
        
        # 应用自定义参数
        if custom_params:
            experiment_config = self.config_loader.merge_configs(experiment_config, custom_params)
        
        # 如果启用故障注入，更新训练配置
        if fault_type:
            experiment_config["training"]["fault_injection"]["enabled"] = True
            experiment_config["training"]["fault_injection"]["fault_types"] = [fault_type]
        
        return experiment_config
    
    def get_fault_injection_schedule(self, fault_type: str) -> Dict[str, Any]:
        """
        获取指定故障类型的注入计划
        
        Args:
            fault_type: 故障类型
            
        Returns:
            注入计划配置
        """
        fault_config = self.config_loader.load_fault_injection_config()
        
        # 查找基于步数的注入计划
        step_based = fault_config.get("injection_schedule", {}).get("step_based", [])
        time_based = fault_config.get("injection_schedule", {}).get("time_based", [])
        
        schedule = {
            "step_based": [s for s in step_based if s.get("fault_type") == fault_type],
            "time_based": [t for t in time_based if t.get("fault_type") == fault_type]
        }
        
        return schedule
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """
        获取监控指标配置
        
        Returns:
            监控指标配置
        """
        monitoring_config = self.config_loader.load_monitoring_config()
        
        metrics = {
            "gpu": [],
            "system": [],
            "training": []
        }
        
        # GPU 指标
        gpu_config = monitoring_config.get("gpu_monitoring", {})
        if gpu_config.get("enabled", False):
            gpu_metrics = gpu_config.get("metrics", {})
            metrics["gpu"] = [name for name, config in gpu_metrics.items() 
                            if config.get("enabled", False)]
        
        # 系统指标
        system_config = monitoring_config.get("system_monitoring", {})
        if system_config.get("enabled", False):
            for category in ["cpu", "memory", "disk", "network"]:
                category_config = system_config.get(category, {})
                if category_config.get("enabled", False):
                    metrics["system"].extend(category_config.get("metrics", []))
        
        # 训练指标
        log_config = monitoring_config.get("log_monitoring", {})
        if log_config.get("enabled", False):
            parsing_rules = log_config.get("parsing_rules", {})
            metrics["training"] = list(parsing_rules.keys())
        
        return metrics


# 全局配置加载器实例
config_loader = ConfigLoader()
experiment_manager = ExperimentConfigManager(config_loader)


def load_config(config_name: str) -> Dict[str, Any]:
    """便捷函数：加载配置文件"""
    return config_loader.load_config(config_name)


def get_experiment_config(experiment_name: str, fault_type: str = None) -> Dict[str, Any]:
    """便捷函数：获取实验配置"""
    return experiment_manager.create_experiment_config(experiment_name, fault_type)


if __name__ == "__main__":
    # 测试配置加载
    try:
        # 加载所有配置文件
        training_config = config_loader.load_training_config()
        fault_config = config_loader.load_fault_injection_config()
        monitoring_config = config_loader.load_monitoring_config()
        
        print("✅ 所有配置文件加载成功!")
        print(f"训练配置: {len(training_config)} 个配置项")
        print(f"故障注入配置: {len(fault_config)} 个配置项")
        print(f"监控配置: {len(monitoring_config)} 个配置项")
        
        # 创建示例实验配置
        exp_config = experiment_manager.create_experiment_config(
            "test_experiment", 
            fault_type="nan_loss"
        )
        print(f"实验配置创建成功: {exp_config['experiment_name']}")
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")