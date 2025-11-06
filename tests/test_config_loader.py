"""
配置加载器单元测试
"""

import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path
import yaml

from src.utils.config_loader import ConfigLoader, ExperimentConfigManager


class TestConfigLoader:
    """配置加载器测试类"""
    
    def setup_method(self):
        """测试方法前置设置"""
        # 创建临时目录和测试配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir()
        
        # 创建测试配置文件
        self.create_test_configs()
        
        # 创建配置加载器实例
        self.loader = ConfigLoader(config_dir=str(self.config_dir))
    
    def teardown_method(self):
        """测试方法后置清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_configs(self):
        """创建测试配置文件"""
        # 训练配置
        training_config = {
            "model_name": "bert-base-uncased",
            "output_dir": "./output",
            "logging_dir": "./logs",
            "max_seq_length": 512,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "weight_decay": 0.01
        }
        
        # 故障注入配置
        fault_config = {
            "injection_schedule": {
                "step_based": [
                    {
                        "fault_type": "nan_loss",
                        "injection_step": 100,
                        "duration": 5
                    }
                ],
                "time_based": [
                    {
                        "fault_type": "oom",
                        "delay": 60,
                        "duration": 30
                    }
                ]
            }
        }
        
        # 监控配置
        monitoring_config = {
            "gpu_monitoring": {
                "enabled": True,
                "metrics": {
                    "utilization_gpu": {"enabled": True},
                    "utilization_memory": {"enabled": True},
                    "memory_used": {"enabled": True}
                }
            },
            "system_monitoring": {
                "enabled": True,
                "cpu": {
                    "enabled": True,
                    "metrics": ["cpu_percent", "load_average"]
                }
            }
        }
        
        # 保存配置文件
        with open(self.config_dir / "training_config.yaml", 'w') as f:
            yaml.dump(training_config, f)
        
        with open(self.config_dir / "fault_injection_config.yaml", 'w') as f:
            yaml.dump(fault_config, f)
        
        with open(self.config_dir / "monitoring_config.yaml", 'w') as f:
            yaml.dump(monitoring_config, f)
    
    def test_load_config_success(self):
        """测试成功加载配置文件"""
        config = self.loader.load_config("training_config")
        assert config is not None
        assert "model_name" in config
        assert config["model_name"] == "bert-base-uncased"
        assert config["learning_rate"] == 2e-5
    
    def test_load_config_file_not_found(self):
        """测试加载不存在的配置文件"""
        with pytest.raises(FileNotFoundError):
            self.loader.load_config("nonexistent_config")
    
    def test_load_invalid_yaml(self):
        """测试加载无效的YAML文件"""
        # 创建无效的YAML文件
        invalid_file = self.config_dir / "invalid_config.yaml"
        with open(invalid_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ValueError):
            self.loader.load_config("invalid_config")
    
    def test_load_training_config(self):
        """测试加载训练配置"""
        config = self.loader.load_training_config()
        assert config is not None
        assert "model_name" in config
        assert "learning_rate" in config
    
    def test_load_fault_injection_config(self):
        """测试加载故障注入配置"""
        config = self.loader.load_fault_injection_config()
        assert config is not None
        assert "injection_schedule" in config
        assert "step_based" in config["injection_schedule"]
    
    def test_load_monitoring_config(self):
        """测试加载监控配置"""
        config = self.loader.load_monitoring_config()
        assert config is not None
        assert "gpu_monitoring" in config
        assert "system_monitoring" in config
    
    def test_get_config_value(self):
        """测试获取嵌套配置值"""
        config = {
            "model": {
                "name": "bert",
                "params": {
                    "hidden_size": 768
                }
            }
        }
        
        # 测试正常路径
        value = self.loader.get_config_value(config, "model.name")
        assert value == "bert"
        
        value = self.loader.get_config_value(config, "model.params.hidden_size")
        assert value == 768
        
        # 测试不存在的路径
        value = self.loader.get_config_value(config, "model.nonexistent")
        assert value is None
        
        # 测试默认值
        value = self.loader.get_config_value(config, "model.nonexistent", "default")
        assert value == "default"
    
    def test_merge_configs(self):
        """测试合并配置"""
        base_config = {
            "model": "bert",
            "learning_rate": 2e-5,
            "params": {
                "hidden_size": 768,
                "num_layers": 12
            }
        }
        
        override_config = {
            "learning_rate": 3e-5,
            "params": {
                "hidden_size": 1024
            },
            "new_param": "value"
        }
        
        merged = self.loader.merge_configs(base_config, override_config)
        
        assert merged["model"] == "bert"  # 保持不变
        assert merged["learning_rate"] == 3e-5  # 被覆盖
        assert merged["params"]["hidden_size"] == 1024  # 嵌套覆盖
        assert merged["params"]["num_layers"] == 12  # 嵌套保持不变
        assert merged["new_param"] == "value"  # 新增参数
    
    def test_validate_config(self):
        """测试配置验证"""
        config = {
            "model": {
                "name": "bert",
                "params": {
                    "hidden_size": 768
                }
            },
            "training": {
                "batch_size": 32
            }
        }
        
        # 测试有效配置
        required_keys = ["model.name", "training.batch_size"]
        assert self.loader.validate_config(config, required_keys) is True
        
        # 测试无效配置
        required_keys = ["model.name", "nonexistent.key"]
        assert self.loader.validate_config(config, required_keys) is False
    
    def test_load_config_with_fallback(self):
        """测试带回退的配置加载"""
        # 测试正常加载
        config = self.loader.load_config_with_fallback("training_config")
        assert config is not None
        
        # 测试回退加载
        fallback_config = {"fallback": "config"}
        config = self.loader.load_config_with_fallback("nonexistent", fallback_config)
        assert config == fallback_config
        
        # 测试默认回退
        config = self.loader.load_config_with_fallback("nonexistent")
        assert config is not None
        assert "model_name" in config  # 应该使用默认回退配置


class TestExperimentConfigManager:
    """实验配置管理器测试类"""
    
    def setup_method(self):
        """测试方法前置设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir()
        
        # 创建测试配置文件
        self.create_test_configs()
        
        # 创建配置管理器和加载器
        self.loader = ConfigLoader(config_dir=str(self.config_dir))
        self.manager = ExperimentConfigManager(self.loader)
    
    def teardown_method(self):
        """测试方法后置清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_configs(self):
        """创建测试配置文件"""
        # 简化版配置文件用于测试
        training_config = {
            "model_name": "test-model",
            "learning_rate": 1e-4
        }
        
        fault_config = {
            "injection_schedule": {
                "step_based": [
                    {"fault_type": "nan_loss", "injection_step": 50}
                ]
            }
        }
        
        monitoring_config = {
            "gpu_monitoring": {"enabled": True},
            "system_monitoring": {"enabled": True}
        }
        
        with open(self.config_dir / "training_config.yaml", 'w') as f:
            yaml.dump(training_config, f)
        
        with open(self.config_dir / "fault_injection_config.yaml", 'w') as f:
            yaml.dump(fault_config, f)
        
        with open(self.config_dir / "monitoring_config.yaml", 'w') as f:
            yaml.dump(monitoring_config, f)
    
    def test_create_experiment_config(self):
        """测试创建实验配置"""
        config = self.manager.create_experiment_config(
            "test_experiment",
            fault_type="nan_loss"
        )
        
        assert config["experiment_name"] == "test_experiment"
        assert config["training"]["model_name"] == "test-model"
        assert config["fault_injection"]["enabled"] is True
        assert config["fault_injection"]["fault_type"] == "nan_loss"
    
    def test_create_experiment_config_with_custom_params(self):
        """测试使用自定义参数创建实验配置"""
        custom_params = {
            "training": {
                "learning_rate": 5e-5
            },
            "custom_setting": "value"
        }
        
        config = self.manager.create_experiment_config(
            "test_experiment",
            custom_params=custom_params
        )
        
        assert config["training"]["learning_rate"] == 5e-5
        assert config["custom_setting"] == "value"
    
    def test_get_fault_injection_schedule(self):
        """测试获取故障注入计划"""
        schedule = self.manager.get_fault_injection_schedule("nan_loss")
        
        assert "step_based" in schedule
        assert "time_based" in schedule
        assert len(schedule["step_based"]) > 0
        assert schedule["step_based"][0]["fault_type"] == "nan_loss"
    
    def test_get_monitoring_metrics(self):
        """测试获取监控指标"""
        metrics = self.manager.get_monitoring_metrics()
        
        assert "gpu" in metrics
        assert "system" in metrics
        assert "training" in metrics
        assert len(metrics["gpu"]) > 0
        assert len(metrics["system"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])