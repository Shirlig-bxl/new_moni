# MONI 系统 API 文档

## 概述

MONI (主动故障注入与多源异常数据收集系统) 是一个完整的"观测器-执行器-注入器"系统，用于主动制造并捕获多源异常，生成带有精确Ground Truth标注的数据集。

## 核心模块

### 1. 配置管理模块

#### ConfigLoader 类

主要功能：加载和管理YAML配置文件。

```python
from src.utils.config_loader import ConfigLoader

# 初始化配置加载器
loader = ConfigLoader(config_dir="./configs")

# 加载特定配置文件
training_config = loader.load_config("training_config")
fault_config = loader.load_config("fault_injection_config")
monitoring_config = loader.load_config("monitoring_config")

# 便捷方法
training_config = loader.load_training_config()
fault_config = loader.load_fault_injection_config()
monitoring_config = loader.load_monitoring_config()

# 获取嵌套配置值
model_name = loader.get_config_value(training_config, "model.name", "bert-base-uncased")

# 合并配置
merged_config = loader.merge_configs(base_config, override_config)

# 验证配置
is_valid = loader.validate_config(config, ["model.name", "learning_rate"])

# 带回退的配置加载
config = loader.load_config_with_fallback("training_config", fallback_config)
```

#### ExperimentConfigManager 类

主要功能：管理实验配置。

```python
from src.utils.config_loader import ExperimentConfigManager

# 初始化配置管理器
manager = ExperimentConfigManager()

# 创建实验配置
experiment_config = manager.create_experiment_config(
    experiment_name="test_experiment",
    fault_type="nan_loss",
    custom_params={"learning_rate": 5e-5}
)

# 获取故障注入计划
schedule = manager.get_fault_injection_schedule("nan_loss")

# 获取监控指标
metrics = manager.get_monitoring_metrics()
```

### 2. 故障注入模块

#### FaultInjector 类

主要功能：在训练过程中主动触发各种故障。

```python
from src.injector.fault_injector import FaultInjector

# 初始化故障注入器
injector = FaultInjector(enable_training_hooks=True)

# 调度故障
injector.schedule_fault(
    fault_type="io_stress",
    delay=60,  # 60秒后触发
    duration=30,  # 持续30秒
    target_dir="/tmp"
)

injector.schedule_fault(
    fault_type="nan_loss",
    delay=120,
    corruption_probability=0.5
)

# 启动故障注入器
injector.start()

# 使用上下文管理器
with injector:
    # 执行训练
    trainer.train()

# 手动停止
injector.stop()
```

#### 支持的故障类型

- **io_stress**: I/O压力测试
- **resource_competition**: 资源竞争（GPU/CPU/内存）
- **process_kill**: 进程终止
- **nan_loss**: NaN Loss注入
- **oom**: 内存溢出注入
- **non_convergence**: 不收敛故障

### 3. 训练钩子模块

#### TrainingFaultInjector 类

主要功能：集成到Hugging Face Transformers Trainer的故障注入回调。

```python
from src.injector.training_hook import TrainingFaultInjector, NaNLossHook, OOMHook

# 创建故障注入器回调
fault_injector = TrainingFaultInjector()

# 添加故障钩子
nan_hook = NaNLossHook(
    trigger_step=100,  # 在第100步触发
    corruption_probability=0.5,
    duration_steps=5
)
fault_injector.add_hook(nan_hook)

oom_hook = OOMHook(
    trigger_step=200,
    memory_size_mb=1024
)
fault_injector.add_hook(oom_hook)

# 在Trainer中使用
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[fault_injector]  # 添加故障注入回调
)
```

#### 可用的故障钩子

- **NaNLossHook**: 注入NaN Loss和梯度损坏
- **OOMHook**: 注入内存溢出故障
- **NonConvergenceHook**: 注入不收敛故障（学习率异常）
- **GradientExplosionHook**: 注入梯度爆炸故障

### 4. 训练执行模块

#### 主训练函数

```python
from src.executor.train import main, main_extended

# 基本训练（BERT微调）
results = main(config_name="baseline", config_path="./configs/training_config.yaml")

# 扩展训练（支持多种模式）
results = main_extended(
    config_name="fault_injection",
    training_mode="bert"  # 或 "gpu_intensive"
)
```

#### GPUTrainingMode 类

主要功能：GPU密集型训练模式。

```python
from src.executor.train import GPUTrainingMode

# 初始化GPU训练器
gpu_trainer = GPUTrainingMode()

# 运行GPU密集型训练
gpu_trainer.run_intensive_training()
```

### 5. 数据聚合模块

#### DataAggregator 类

主要功能：聚合多源监控数据。

```python
from src.aggregator.data_aggregator import DataAggregator

# 初始化数据聚合器
aggregator = DataAggregator(
    time_granularity=1,  # 1秒时间粒度
    use_parallel=True,   # 启用并行处理
    chunk_size=10000,    # 分块大小
    optimize_memory=True # 内存优化
)

# 聚合数据
aggregated_data = aggregator.aggregate_data(
    gpu_file="gpu_metrics.csv",
    system_file="system_metrics.csv", 
    training_file="training_metrics.csv"
)

# 保存结果
aggregator.save_aggregated_data("aggregated_data.csv")

# 获取数据质量报告
quality_report = aggregator.get_data_quality_report()
```

#### TSEMatrixBuilder 类

主要功能：构建TSE-Matrix和Ground Truth标注。

```python
from src.aggregator.tse_matrix_builder import TSEMatrixBuilder

# 初始化TSE-Matrix构建器
builder = TSEMatrixBuilder()

# 构建TSE-Matrix
tse_matrix, ground_truth = builder.build_tse_matrix(
    aggregated_data=aggregated_data,
    fault_config=fault_config,
    experiment_start_time=experiment_start_time
)

# 保存结果
builder.save_tse_matrix("./output", "experiment_1")

# 获取统计信息
stats = builder.get_statistics()
```

### 6. 监控模块

#### GPU监控

```python
from src.observer.gpu_monitor import GPUMonitor

# 初始化GPU监控器
gpu_monitor = GPUMonitor(interval=1)  # 1秒间隔

# 开始监控
gpu_monitor.start_monitoring()

# 停止监控并保存数据
gpu_data = gpu_monitor.stop_monitoring()
gpu_monitor.save_to_csv("gpu_metrics.csv")
```

#### 系统监控

```python
from src.observer.system_monitor import SystemMonitor

# 初始化系统监控器
system_monitor = SystemMonitor(interval=1)

# 开始监控
system_monitor.start_monitoring()

# 停止监控并保存数据
system_data = system_monitor.stop_monitoring()
system_monitor.save_to_csv("system_metrics.csv")
```

## 配置示例

### 训练配置 (configs/training_config.yaml)

```yaml
model_name: "bert-base-uncased"
output_dir: "./output"
logging_dir: "./logs"
max_seq_length: 512
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
num_train_epochs: 3
learning_rate: 2e-5
weight_decay: 0.01
```

### 故障注入配置 (configs/fault_injection_config.yaml)

```yaml
injection_schedule:
  step_based:
    - fault_type: "nan_loss"
      injection_step: 100
      duration: 5
    - fault_type: "oom" 
      injection_step: 200
      duration: 10
  time_based:
    - fault_type: "io_stress"
      delay: 60
      duration: 30
```

### 监控配置 (configs/monitoring_config.yaml)

```yaml
gpu_monitoring:
  enabled: true
  metrics:
    utilization_gpu:
      enabled: true
    utilization_memory:
      enabled: true
    memory_used:
      enabled: true

system_monitoring:
  enabled: true
  cpu:
    enabled: true
    metrics:
      - cpu_percent
      - load_average
  memory:
    enabled: true
    metrics:
      - memory_percent
      - memory_used
```

## 命令行工具

### 运行完整实验

```bash
python run_local_experiment.py
```

### 单独运行训练

```bash
python -m src.executor.train --config-name baseline --training-mode bert
```

### 单独运行数据聚合

```bash
python -m src.aggregator.data_aggregator \
  --gpu-file gpu_metrics.csv \
  --system-file system_metrics.csv \
  --training-file training_metrics.csv \
  --output aggregated_data.csv
```

### 构建TSE-Matrix

```bash
python -m src.aggregator.tse_matrix_builder \
  --gpu-file gpu_metrics.csv \
  --system-file system_metrics.csv \
  --training-file training_metrics.csv \
  --output-dir ./output \
  --experiment-name experiment_1
```

## 测试

运行单元测试：

```bash
pytest tests/ -v
```

运行特定测试模块：

```bash
pytest tests/test_config_loader.py -v
pytest tests/test_fault_injector.py -v
```

## 错误处理

系统包含完善的错误处理机制：

- 配置加载失败时的回退配置
- 数据加载失败时的模拟数据生成
- 训练过程中的异常检测和日志记录
- 内存优化和并行处理错误处理

## 性能优化

- **并行处理**: 支持多进程数据加载和处理
- **内存优化**: 自动优化DataFrame内存使用
- **分块处理**: 支持大数据集的分块处理
- **增量处理**: 支持增量数据聚合

## 许可证

MIT License