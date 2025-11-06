"""
测试大型数据集管理器
验证大数据集生成和存储功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.large_dataset_manager import LargeDatasetManager, create_multi_model_dataset_manager
from experiments.multi_model_orchestrator import MultiModelOrchestrator


def test_large_dataset_manager():
    """测试大型数据集管理器基本功能"""
    print("=" * 60)
    print("测试大型数据集管理器")
    print("=" * 60)
    
    try:
        # 创建数据集管理器
        manager = create_multi_model_dataset_manager()
        print("✅ 大型数据集管理器创建成功")
        
        # 创建测试数据
        print("\n1. 创建测试数据...")
        test_data = create_test_data(1000)
        print(f"   测试数据创建完成: {len(test_data)} 条记录")
        
        # 添加数据到数据集
        print("\n2. 添加数据到数据集...")
        success = manager.add_data_to_dataset("multi_model_tse_matrix", test_data)
        print(f"   数据添加: {'✅ 成功' if success else '❌ 失败'}")
        
        # 查询数据集信息
        print("\n3. 查询数据集信息...")
        dataset_info = manager.get_dataset_info("multi_model_tse_matrix")
        print(f"   数据集信息:")
        print(f"   - 总记录数: {dataset_info.get('total_records', 0)}")
        print(f"   - 总大小: {dataset_info.get('total_size_mb', 0):.2f} MB")
        print(f"   - 批次数量: {dataset_info.get('batch_count', 0)}")
        
        # 查询数据
        print("\n4. 查询数据...")
        query_result = manager.query_dataset("multi_model_tse_matrix", limit=5)
        print(f"   查询结果: {len(query_result)} 条记录")
        if not query_result.empty:
            print("   前5条记录:")
            for i, (_, row) in enumerate(query_result.head().iterrows()):
                print(f"     {i+1}. 时间: {row['timestamp']}, 异常: {row['is_anomaly']}")
        
        # 导出数据集
        print("\n5. 导出数据集...")
        export_file = manager.export_dataset("multi_model_tse_matrix", "parquet")
        print(f"   数据集导出: {'✅ 成功' if export_file else '❌ 失败'}")
        if export_file:
            print(f"   导出文件: {export_file}")
        
        print("\n✅ 大型数据集管理器测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_model_orchestrator_integration():
    """测试多模型实验编排器与大型数据集管理器的集成"""
    print("\n" + "=" * 60)
    print("测试多模型实验编排器集成")
    print("=" * 60)
    
    try:
        # 创建编排器（启用大型数据集管理器）
        orchestrator = MultiModelOrchestrator(
            output_dir="./test_integration",
            use_large_dataset_manager=True
        )
        print("✅ 多模型实验编排器创建成功")
        
        # 创建演示实验计划
        print("\n1. 创建演示实验计划...")
        demo_config = create_demo_experiment_plan()
        plan = orchestrator.create_experiment_plan(
            plan_name="test_integration_plan",
            model_configs=demo_config["model_configs"],
            dataset_configs=demo_config["dataset_configs"],
            fault_configs=demo_config["fault_configs"],
            training_config=demo_config["training_config"]
        )
        print(f"   实验计划创建: ✅ 成功")
        print(f"   总实验数: {plan['total_experiments']}")
        
        # 获取数据集统计信息
        print("\n2. 获取数据集统计信息...")
        stats = orchestrator.get_dataset_statistics()
        if "error" not in stats:
            print(f"   数据集统计:")
            print(f"   - 总记录数: {stats.get('total_records', 0)}")
            print(f"   - 批次数量: {stats.get('batch_count', 0)}")
        else:
            print(f"   数据集统计: {stats['error']}")
        
        # 测试数据查询功能
        print("\n3. 测试数据查询功能...")
        query_result = orchestrator.query_experiment_data(limit=3)
        print(f"   数据查询: {len(query_result)} 条记录")
        
        print("\n✅ 多模型实验编排器集成测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_data(num_records: int) -> pd.DataFrame:
    """创建测试数据"""
    timestamps = pd.date_range('2024-01-01', periods=num_records, freq='1S')
    
    data = {
        'timestamp': timestamps,
        'is_anomaly': np.random.randint(0, 2, num_records),
        'gpu_utilization_gpu': np.random.uniform(0, 100, num_records),
        'gpu_utilization_memory': np.random.uniform(0, 100, num_records),
        'gpu_memory_total': np.full(num_records, 16000),
        'gpu_memory_used': np.random.randint(0, 16000, num_records),
        'gpu_memory_free': np.random.randint(0, 16000, num_records),
        'gpu_temperature': np.random.uniform(30, 85, num_records),
        'gpu_power_draw': np.random.uniform(50, 300, num_records),
        'sys_cpu_percent': np.random.uniform(0, 100, num_records),
        'sys_memory_percent': np.random.uniform(0, 100, num_records),
        'sys_disk_usage': np.random.uniform(0, 100, num_records),
        'train_step': np.arange(num_records),
        'train_loss': np.random.uniform(0, 2, num_records),
        'train_learning_rate': np.full(num_records, 2e-5),
        'train_throughput': np.random.uniform(10, 1000, num_records),
        'train_step_time': np.random.uniform(0.1, 1.0, num_records),
        'eval_accuracy': np.random.uniform(0.7, 0.95, num_records),
        'eval_loss': np.random.uniform(0.1, 1.0, num_records),
        'event_anomaly': np.random.randint(0, 2, num_records),
        'gpu_memory_utilization_ratio': np.random.uniform(0, 1, num_records),
        'train_step_rate': np.random.uniform(0.5, 2.0, num_records),
        'train_loss_change_rate': np.random.uniform(-0.1, 0.1, num_records),
        'total_anomaly_events': np.random.randint(0, 5, num_records)
    }
    
    return pd.DataFrame(data)


def create_demo_experiment_plan() -> dict:
    """创建演示实验计划配置"""
    
    # 模型配置
    model_configs = [
        {"model_type": "bert_classification", "model_name": "bert-base-uncased"},
        {"model_type": "distilbert", "model_name": "distilbert-base-uncased"}
    ]
    
    # 数据集配置
    dataset_configs = [
        {"dataset_type": "imdb_sentiment", "split_sizes": {"train": 100, "test": 20}},
        {"dataset_type": "glue_sst2", "split_sizes": {"train": 100, "test": 20}}
    ]
    
    # 故障配置
    fault_configs = [
        {"enabled": False, "fault_type": "baseline"},
        {"enabled": True, "fault_type": "nan_loss", 
         "injection_schedule": [{"fault_type": "nan_loss", "delay": 30, "duration": 10}]}
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


if __name__ == "__main__":
    print("开始测试大数据集生成和存储功能...")
    
    # 测试大型数据集管理器
    test1_success = test_large_dataset_manager()
    
    # 测试多模型实验编排器集成
    test2_success = test_multi_model_orchestrator_integration()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"大型数据集管理器测试: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"多模型编排器集成测试: {'✅ 通过' if test2_success else '❌ 失败'}")
    
    if test1_success and test2_success:
        print("\n🎉 所有测试通过! 大数据集生成和存储功能正常工作。")
    else:
        print("\n⚠️  部分测试失败，请检查相关代码。")