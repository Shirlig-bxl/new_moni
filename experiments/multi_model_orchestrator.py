"""
多模型实验编排器
协调多个模型和数据集的大规模实验，生成大规模异常检测数据集
集成大型数据集管理器优化数据存储和检索
"""

import os
import time
import logging
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.executor.model_factory import ModelFactory, ModelType, MultiModelTrainer
from src.executor.dataset_manager import DatasetManager, DatasetType, TaskType
from src.injector.fault_injector import FaultInjector
from src.aggregator.data_aggregator import DataAggregator
from src.aggregator.tse_matrix_builder import TSEMatrixBuilder
from src.utils.config_loader import ConfigLoader
from src.utils.large_dataset_manager import LargeDatasetManager, create_multi_model_dataset_manager

logger = logging.getLogger(__name__)


class MultiModelOrchestrator:
    """多模型实验编排器"""
    
    def __init__(self, output_dir: str = "./multi_model_experiments", use_large_dataset_manager: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_factory = ModelFactory()
        self.dataset_manager = DatasetManager()
        self.fault_injector = FaultInjector(enable_training_hooks=True)
        self.config_loader = ConfigLoader()
        
        # 集成大型数据集管理器
        self.use_large_dataset_manager = use_large_dataset_manager
        if use_large_dataset_manager:
            self.dataset_storage_manager = create_multi_model_dataset_manager()
            logger.info("大型数据集管理器已集成")
        else:
            self.dataset_storage_manager = None
        
        self.experiment_results = {}
        self.experiment_metadata = {}
        
        logger.info(f"多模型实验编排器初始化完成，输出目录: {output_dir}, 大型数据集管理: {use_large_dataset_manager}")
    
    def create_experiment_plan(self, plan_name: str, 
                             model_configs: List[Dict[str, Any]],
                             dataset_configs: List[Dict[str, Any]],
                             fault_configs: List[Dict[str, Any]],
                             training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建实验计划
        
        Args:
            plan_name: 计划名称
            model_configs: 模型配置列表
            dataset_configs: 数据集配置列表
            fault_configs: 故障配置列表
            training_config: 训练配置
            
        Returns:
            实验计划配置
        """
        experiment_plan = {
            "plan_name": plan_name,
            "created_time": datetime.now().isoformat(),
            "total_experiments": len(model_configs) * len(dataset_configs) * len(fault_configs),
            "model_configs": model_configs,
            "dataset_configs": dataset_configs,
            "fault_configs": fault_configs,
            "training_config": training_config,
            "experiment_combinations": []
        }
        
        # 生成所有实验组合
        for model_config in model_configs:
            for dataset_config in dataset_configs:
                for fault_config in fault_configs:
                    experiment_id = f"{model_config['model_name']}_{dataset_config['dataset_type']}_{fault_config.get('fault_type', 'baseline')}"
                    
                    combination = {
                        "experiment_id": experiment_id,
                        "model_config": model_config,
                        "dataset_config": dataset_config,
                        "fault_config": fault_config,
                        "status": "pending",
                        "start_time": None,
                        "end_time": None,
                        "results": None
                    }
                    experiment_plan["experiment_combinations"].append(combination)
        
        # 保存实验计划
        plan_file = self.output_dir / f"{plan_name}_plan.yaml"
        with open(plan_file, 'w', encoding='utf-8') as f:
            yaml.dump(experiment_plan, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"创建实验计划: {plan_name}, 包含 {experiment_plan['total_experiments']} 个实验")
        return experiment_plan
    
    def run_experiment_plan(self, plan_name: str, max_workers: int = 4, 
                          skip_completed: bool = True) -> Dict[str, Any]:
        """
        运行实验计划
        
        Args:
            plan_name: 计划名称
            max_workers: 最大工作线程数
            skip_completed: 是否跳过已完成的实验
            
        Returns:
            实验结果汇总
        """
        # 加载实验计划
        plan_file = self.output_dir / f"{plan_name}_plan.yaml"
        if not plan_file.exists():
            raise FileNotFoundError(f"实验计划文件不存在: {plan_file}")
        
        with open(plan_file, 'r', encoding='utf-8') as f:
            experiment_plan = yaml.safe_load(f)
        
        logger.info(f"开始运行实验计划: {plan_name}")
        
        # 过滤需要运行的实验
        experiments_to_run = []
        for combination in experiment_plan["experiment_combinations"]:
            if skip_completed and combination["status"] == "completed":
                continue
            experiments_to_run.append(combination)
        
        logger.info(f"需要运行的实验数量: {len(experiments_to_run)}")
        
        # 并行运行实验
        results = self._run_experiments_parallel(experiments_to_run, max_workers)
        
        # 更新实验计划
        self._update_experiment_plan(experiment_plan, results)
        
        # 保存更新后的计划
        with open(plan_file, 'w', encoding='utf-8') as f:
            yaml.dump(experiment_plan, f, default_flow_style=False, allow_unicode=True)
        
        # 生成实验报告
        report = self._generate_experiment_report(experiment_plan)
        
        logger.info(f"实验计划完成: {plan_name}")
        return report
    
    def _run_experiments_parallel(self, experiments: List[Dict[str, Any]], 
                                max_workers: int) -> Dict[str, Any]:
        """并行运行多个实验"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_exp = {
                executor.submit(self._run_single_experiment, exp): exp 
                for exp in experiments
            }
            
            for future in as_completed(future_to_exp):
                experiment = future_to_exp[future]
                try:
                    result = future.result()
                    results[experiment["experiment_id"]] = result
                    logger.info(f"实验完成: {experiment['experiment_id']}")
                except Exception as e:
                    logger.error(f"实验失败: {experiment['experiment_id']}, 错误: {e}")
                    results[experiment["experiment_id"]] = {
                        "success": False,
                        "error": str(e)
                    }
        
        return results
    
    def _run_single_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个实验"""
        experiment_id = experiment["experiment_id"]
        experiment_dir = self.output_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        logger.info(f"开始实验: {experiment_id}")
        
        start_time = time.time()
        experiment["start_time"] = datetime.now().isoformat()
        
        try:
            # 加载数据集
            dataset_config = experiment["dataset_config"]
            dataset_type = DatasetType(dataset_config["dataset_type"])
            dataset = self.dataset_manager.load_dataset(
                dataset_type, 
                split_sizes=dataset_config.get("split_sizes", {"train": 1000, "test": 200})
            )
            
            # 创建模型
            model_config = experiment["model_config"]
            model_type = ModelType(model_config["model_type"])
            model_name = model_config["model_name"]
            
            num_labels = dataset_config.get("num_labels")
            if num_labels is None and dataset_type in self.dataset_manager.DATASET_CONFIGS:
                num_labels = self.dataset_manager.DATASET_CONFIGS[dataset_type].get("num_labels")
            
            model, tokenizer = self.model_factory.create_model(
                model_type, model_name, num_labels=num_labels
            )
            
            # 预处理数据集
            processed_dataset = self.dataset_manager.preprocess_dataset(
                dataset, dataset_type, tokenizer=tokenizer
            )
            
            # 配置故障注入
            fault_config = experiment["fault_config"]
            if fault_config.get("enabled", False):
                fault_injector = self._setup_fault_injection(fault_config)
            else:
                fault_injector = None
            
            # 训练模型（这里简化实现，实际应该调用完整的训练流程）
            training_results = self._train_model_simplified(
                model, processed_dataset, experiment["training_config"], fault_injector
            )
            
            # 收集监控数据
            monitoring_data = self._collect_monitoring_data(experiment_dir)
            
            # 聚合数据
            aggregated_data = self._aggregate_experiment_data(
                training_results, monitoring_data, experiment_dir
            )
            
            # 构建TSE-Matrix
            tse_matrix, ground_truth = self._build_tse_matrix(
                aggregated_data, fault_config, experiment_dir
            )

            # 使用大型数据集管理器存储数据
            if self.use_large_dataset_manager and tse_matrix is not None:
                storage_result = self._store_in_large_dataset_manager(
                    experiment_id, tse_matrix, ground_truth, experiment
                )
            else:
                storage_result = {"success": False, "reason": "large_dataset_manager_disabled"}

            end_time = time.time()
            experiment_duration = end_time - start_time
            
            result = {
                "success": True,
                "experiment_id": experiment_id,
                "duration": experiment_duration,
                "training_results": training_results,
                "monitoring_data": monitoring_data,
                "tse_matrix_shape": tse_matrix.shape if tse_matrix is not None else None,
                "anomaly_count": int(ground_truth.sum()) if ground_truth is not None else 0,
                "output_files": self._get_output_files(experiment_dir),
                "large_dataset_storage": storage_result
            }
            
            experiment["status"] = "completed"
            experiment["end_time"] = datetime.now().isoformat()
            experiment["results"] = result
            
            # 保存实验结果
            result_file = experiment_dir / "experiment_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"实验成功: {experiment_id}, 耗时: {experiment_duration:.2f}s")
            return result
            
        except Exception as e:
            end_time = time.time()
            experiment_duration = end_time - start_time
            
            error_result = {
                "success": False,
                "experiment_id": experiment_id,
                "duration": experiment_duration,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            experiment["status"] = "failed"
            experiment["end_time"] = datetime.now().isoformat()
            experiment["results"] = error_result
            
            # 保存错误结果
            result_file = experiment_dir / "experiment_error.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            
            logger.error(f"实验失败: {experiment_id}, 错误: {e}")
            return error_result
    
    def _setup_fault_injection(self, fault_config: Dict[str, Any]) -> FaultInjector:
        """设置故障注入"""
        injector = FaultInjector(enable_training_hooks=True)
        
        # 配置故障注入计划
        injection_schedule = fault_config.get("injection_schedule", [])
        for injection in injection_schedule:
            injector.schedule_fault(**injection)
        
        return injector
    
    def _train_model_simplified(self, model, dataset, training_config: Dict[str, Any],
                              fault_injector: Optional[FaultInjector]) -> Dict[str, Any]:
        """简化版模型训练（实际项目中应该使用完整的训练流程）"""
        # 这里简化实现，实际应该调用 src.executor.train 中的训练逻辑
        # 为了演示目的，返回模拟的训练结果
        
        import random
        
        training_results = {
            "final_loss": random.uniform(0.1, 1.0),
            "final_accuracy": random.uniform(0.7, 0.95),
            "training_steps": random.randint(100, 1000),
            "training_time": random.uniform(60, 600),
            "memory_peak": random.uniform(100, 5000),
            "fault_injection_activated": fault_injector is not None
        }
        
        # 如果启用了故障注入，模拟故障影响
        if fault_injector and training_config.get("enable_fault_injection", False):
            training_results["fault_effects"] = {
                "nan_loss_occurred": random.random() < 0.3,
                "oom_occurred": random.random() < 0.2,
                "training_interrupted": random.random() < 0.1
            }
        
        return training_results
    
    def _collect_monitoring_data(self, experiment_dir: Path) -> Dict[str, Any]:
        """收集监控数据"""
        # 这里简化实现，实际应该从系统监控、GPU监控等收集真实数据
        
        import psutil
        import torch
        
        monitoring_data = {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "gpu": {
                "available": torch.cuda.is_available(),
                "memory_allocated": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                "memory_reserved": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            },
            "training_metrics": {
                "throughput": random.uniform(10, 1000),
                "batch_times": [random.uniform(0.1, 1.0) for _ in range(10)]
            }
        }
        
        # 保存监控数据
        monitoring_file = experiment_dir / "monitoring_data.json"
        with open(monitoring_file, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2)
        
        return monitoring_data
    
    def _aggregate_experiment_data(self, training_results: Dict[str, Any],
                                 monitoring_data: Dict[str, Any],
                                 experiment_dir: Path) -> pd.DataFrame:
        """聚合实验数据"""
        # 创建模拟的聚合数据
        timestamps = pd.date_range(start=datetime.now(), periods=100, freq='1S')
        
        aggregated_data = []
        for i, timestamp in enumerate(timestamps):
            row = {
                'timestamp': timestamp,
                'training_loss': training_results.get('final_loss', 0) * (0.9 + 0.2 * np.sin(i/10)),
                'training_accuracy': training_results.get('final_accuracy', 0) * (0.95 + 0.1 * np.cos(i/10)),
                'cpu_usage': monitoring_data['system']['cpu_percent'] * (0.8 + 0.4 * np.random.random()),
                'memory_usage': monitoring_data['system']['memory_percent'] * (0.7 + 0.6 * np.random.random()),
                'gpu_memory': monitoring_data['gpu']['memory_allocated'] * (0.5 + np.random.random())
            }
            
            # 添加一些异常事件
            if i % 20 == 0:
                row['event_anomaly'] = 1
                row['anomaly_score'] = np.random.uniform(0.7, 1.0)
            else:
                row['event_anomaly'] = 0
                row['anomaly_score'] = np.random.uniform(0.0, 0.3)
            
            aggregated_data.append(row)
        
        df = pd.DataFrame(aggregated_data)
        
        # 保存聚合数据
        aggregated_file = experiment_dir / "aggregated_data.csv"
        df.to_csv(aggregated_file, index=False)
        
        return df
    
    def _build_tse_matrix(self, aggregated_data: pd.DataFrame, 
                        fault_config: Dict[str, Any],
                        experiment_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
        """构建TSE-Matrix"""
        builder = TSEMatrixBuilder()
        
        # 使用故障配置创建ground truth
        experiment_start_time = aggregated_data['timestamp'].min()
        
        tse_matrix, ground_truth = builder.build_tse_matrix(
            aggregated_data=aggregated_data,
            fault_config=fault_config if fault_config.get("enabled", False) else None,
            experiment_start_time=experiment_start_time
        )
        
        # 保存TSE-Matrix
        builder.save_tse_matrix(str(experiment_dir), "experiment")
        
        return tse_matrix, ground_truth
    
    def _get_output_files(self, experiment_dir: Path) -> List[str]:
        """获取输出文件列表"""
        files = []
        for file_path in experiment_dir.glob("*"):
            if file_path.is_file():
                files.append(file_path.name)
        return files
    
    def _store_in_large_dataset_manager(self,
                                      experiment_id: str,
                                      tse_matrix: pd.DataFrame,
                                      ground_truth: pd.Series,
                                      experiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用大型数据集管理器存储实验数据
        
        Args:
            experiment_id: 实验ID
            tse_matrix: TSE矩阵数据
            ground_truth: 地面真值标签
            experiment: 实验配置
            
        Returns:
            存储结果
        """
        if not self.use_large_dataset_manager or self.dataset_storage_manager is None:
            return {"success": False, "reason": "large_dataset_manager_disabled"}
        
        try:
            # 合并TSE矩阵和地面真值
            combined_data = tse_matrix.copy()
            combined_data['is_anomaly'] = ground_truth
            
            # 添加实验元数据列
            combined_data['experiment_id'] = experiment_id
            combined_data['model_type'] = experiment['model_config']['model_type']
            combined_data['dataset_type'] = experiment['dataset_config']['dataset_type']
            combined_data['fault_type'] = experiment['fault_config'].get('fault_type', 'baseline')
            
            # 使用大型数据集管理器存储数据
            batch_id = f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            success = self.dataset_storage_manager.add_data_to_dataset(
                dataset_name="multi_model_tse_matrix",
                data=combined_data,
                batch_id=batch_id,
                incremental=True
            )
            
            if success:
                # 获取存储信息
                dataset_info = self.dataset_storage_manager.get_dataset_info("multi_model_tse_matrix")
                return {
                    "success": True,
                    "batch_id": batch_id,
                    "dataset_info": dataset_info
                }
            else:
                return {
                    "success": False,
                    "reason": "add_data_to_dataset_failed"
                }
                
        except Exception as e:
            logger.error(f"使用大型数据集管理器存储数据失败: {e}")
            return {
                "success": False,
                "reason": f"exception: {str(e)}"
            }
    
    def query_experiment_data(self,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            model_types: Optional[List[str]] = None,
                            dataset_types: Optional[List[str]] = None,
                            fault_types: Optional[List[str]] = None,
                            anomaly_only: bool = False,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        查询实验数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            model_types: 模型类型过滤
            dataset_types: 数据集类型过滤
            fault_types: 故障类型过滤
            anomaly_only: 是否只查询异常数据
            limit: 限制记录数
            
        Returns:
            查询结果DataFrame
        """
        if not self.use_large_dataset_manager or self.dataset_storage_manager is None:
            logger.error("大型数据集管理器未启用")
            return pd.DataFrame()
        
        try:
            # 构建过滤条件
            filters = {}
            if model_types:
                filters['model_type'] = model_types
            if dataset_types:
                filters['dataset_type'] = dataset_types
            if fault_types:
                filters['fault_type'] = fault_types
            if anomaly_only:
                filters['is_anomaly'] = 1
            
            # 查询数据
            result = self.dataset_storage_manager.query_dataset(
                dataset_name="multi_model_tse_matrix",
                start_time=start_time,
                end_time=end_time,
                filters=filters if filters else None,
                limit=limit
            )
            
            logger.info(f"数据查询完成: {len(result)} 条记录")
            return result
            
        except Exception as e:
            logger.error(f"查询实验数据失败: {e}")
            return pd.DataFrame()
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            数据集统计信息
        """
        if not self.use_large_dataset_manager or self.dataset_storage_manager is None:
            return {"error": "large_dataset_manager_disabled"}
        
        try:
            dataset_info = self.dataset_storage_manager.get_dataset_info("multi_model_tse_matrix")
            return dataset_info
        except Exception as e:
            return {"error": str(e)}
    
    def export_combined_dataset(self,
                              output_format: str = "parquet",
                              output_dir: Optional[str] = None) -> str:
        """
        导出合并的数据集
        
        Args:
            output_format: 输出格式
            output_dir: 输出目录
            
        Returns:
            输出文件路径
        """
        if not self.use_large_dataset_manager or self.dataset_storage_manager is None:
            logger.error("大型数据集管理器未启用")
            return ""
        
        try:
            output_file = self.dataset_storage_manager.export_dataset(
                dataset_name="multi_model_tse_matrix",
                output_format=output_format,
                output_dir=output_dir
            )
            
            logger.info(f"数据集导出成功: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"导出数据集失败: {e}")
            return ""
    
    def _update_experiment_plan(self, experiment_plan: Dict[str, Any], 
                              results: Dict[str, Any]):
        """更新实验计划"""
        for combination in experiment_plan["experiment_combinations"]:
            exp_id = combination["experiment_id"]
            if exp_id in results:
                combination["results"] = results[exp_id]
                combination["status"] = "completed" if results[exp_id]["success"] else "failed"
    
    def _generate_experiment_report(self, experiment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """生成实验报告"""
        total_experiments = len(experiment_plan["experiment_combinations"])
        completed = sum(1 for exp in experiment_plan["experiment_combinations"] if exp["status"] == "completed")
        failed = sum(1 for exp in experiment_plan["experiment_combinations"] if exp["status"] == "failed")
        pending = total_experiments - completed - failed
        
        # 计算统计信息
        successful_results = [
            exp["results"] for exp in experiment_plan["experiment_combinations"] 
            if exp["status"] == "completed" and exp["results"]["success"]
        ]
        
        if successful_results:
            avg_duration = np.mean([res["duration"] for res in successful_results])
            total_anomalies = sum(res.get("anomaly_count", 0) for res in successful_results)
            avg_accuracy = np.mean([res["training_results"].get("final_accuracy", 0) for res in successful_results])
        else:
            avg_duration = 0
            total_anomalies = 0
            avg_accuracy = 0
        
        report = {
            "plan_name": experiment_plan["plan_name"],
            "generated_time": datetime.now().isoformat(),
            "summary": {
                "total_experiments": total_experiments,
                "completed": completed,
                "failed": failed,
                "pending": pending,
                "success_rate": completed / total_experiments if total_experiments > 0 else 0
            },
            "statistics": {
                "average_duration": avg_duration,
                "total_anomalies": total_anomalies,
                "average_accuracy": avg_accuracy,
                "total_tse_matrices": completed
            },
            "model_performance": {},
            "dataset_usage": {},
            "fault_analysis": {}
        }
        
        # 按模型类型分析性能
        model_performance = {}
        for exp in experiment_plan["experiment_combinations"]:
            if exp["status"] == "completed" and exp["results"]["success"]:
                model_type = exp["model_config"]["model_type"]
                if model_type not in model_performance:
                    model_performance[model_type] = []
                
                model_performance[model_type].append(exp["results"]["training_results"]["final_accuracy"])
        
        for model_type, accuracies in model_performance.items():
            report["model_performance"][model_type] = {
                "average_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "count": len(accuracies)
            }
        
        # 保存报告
        report_file = self.output_dir / f"{experiment_plan['plan_name']}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验报告已生成: {report_file}")
        return report


# 便捷函数和示例配置
def create_demo_experiment_plan() -> Dict[str, Any]:
    """创建演示用实验计划"""
    
    # 模型配置
    model_configs = [
        {"model_type": "bert_classification", "model_name": "bert-base-uncased"},
        {"model_type": "distilbert", "model_name": "distilbert-base-uncased"},
        {"model_type": "roberta", "model_name": "roberta-base"}
    ]
    
    # 数据集配置
    dataset_configs = [
        {"dataset_type": "imdb_sentiment", "split_sizes": {"train": 500, "test": 100}},
        {"dataset_type": "glue_sst2", "split_sizes": {"train": 500, "test": 100}}
    ]
    
    # 故障配置
    fault_configs = [
        {"enabled": False, "fault_type": "baseline"},  # 基线，无故障
        {"enabled": True, "fault_type": "nan_loss", 
         "injection_schedule": [{"fault_type": "nan_loss", "delay": 30, "duration": 10}]},
        {"enabled": True, "fault_type": "oom",
         "injection_schedule": [{"fault_type": "oom", "delay": 60, "duration": 20}]}
    ]
    
    # 训练配置
    training_config = {
        "learning_rate": 2e-5,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "enable_fault_injection": True
    }
    
    return {
        "model_configs": model_configs,
        "dataset_configs": dataset_configs,
        "fault_configs": fault_configs,
        "training_config": training_config
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多模型实验编排器")
    parser.add_argument("--plan-name", required=True, help="实验计划名称")
    parser.add_argument("--output-dir", default="./multi_model_experiments", help="输出目录")
    parser.add_argument("--max-workers", type=int, default=2, help="最大工作线程数")
    parser.add_argument("--create-demo", action="store_true", help="创建演示实验计划")
    
    args = parser.parse_args()
    
    # 初始化编排器
    orchestrator = MultiModelOrchestrator(output_dir=args.output_dir)
    
    if args.create_demo:
        # 创建演示实验计划
        demo_config = create_demo_experiment_plan()
        plan = orchestrator.create_experiment_plan(
            plan_name=args.plan_name,
            model_configs=demo_config["model_configs"],
            dataset_configs=demo_config["dataset_configs"],
            fault_configs=demo_config["fault_configs"],
            training_config=demo_config["training_config"]
        )
        print(f"演示实验计划已创建: {args.plan_name}")
        print(f"总实验数: {plan['total_experiments']}")
    
    else:
        # 运行实验计划
        report = orchestrator.run_experiment_plan(
            plan_name=args.plan_name,
            max_workers=args.max_workers
        )
        print(f"实验计划完成: {args.plan_name}")
        print(f"实验结果: {report['summary']}")


if __name__ == "__main__":
    main()