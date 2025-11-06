"""
大型数据集管理器
优化大数据集的生成、存储和检索，支持高效的数据格式和增量处理
使用标准库和pandas内置功能，不依赖外部库
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import gc
import zlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)


class LargeDatasetManager:
    """大型数据集管理器"""
    
    def __init__(self, 
                 storage_dir: str = "./large_datasets",
                 compression: bool = True,
                 chunk_size: int = 10000,
                 max_memory_mb: int = 1024):
        """
        初始化数据集管理器
        
        Args:
            storage_dir: 存储目录
            compression: 是否启用压缩
            chunk_size: 分块大小
            max_memory_mb: 最大内存使用量 (MB)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.compression = compression
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        
        self.dataset_metadata = {}
        self.current_datasets = {}
        
        logger.info(f"大型数据集管理器初始化完成: {storage_dir}")
    
    def create_dataset(self, 
                      dataset_name: str, 
                      schema: Dict[str, str],
                      description: str = "",
                      tags: List[str] = None) -> str:
        """
        创建新数据集
        
        Args:
            dataset_name: 数据集名称
            schema: 数据模式 {列名: 数据类型}
            description: 数据集描述
            tags: 标签列表
            
        Returns:
            数据集路径
        """
        dataset_path = self.storage_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        # 创建数据集元数据
        metadata = {
            "dataset_name": dataset_name,
            "created_time": datetime.now().isoformat(),
            "schema": schema,
            "description": description,
            "tags": tags or [],
            "total_records": 0,
            "total_size_mb": 0,
            "compression": self.compression,
            "chunk_size": self.chunk_size,
            "file_formats": ["csv", "pickle", "metadata"],
            "last_updated": datetime.now().isoformat()
        }
        
        # 保存元数据
        metadata_file = dataset_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 创建数据目录
        (dataset_path / "data").mkdir(exist_ok=True)
        (dataset_path / "indices").mkdir(exist_ok=True)
        (dataset_path / "cache").mkdir(exist_ok=True)
        
        self.dataset_metadata[dataset_name] = metadata
        logger.info(f"数据集创建成功: {dataset_name}")
        
        return str(dataset_path)
    
    def add_data_to_dataset(self, 
                           dataset_name: str, 
                           data: pd.DataFrame,
                           batch_id: str = None,
                           incremental: bool = True) -> bool:
        """
        添加数据到数据集
        
        Args:
            dataset_name: 数据集名称
            data: 要添加的数据
            batch_id: 批次ID
            incremental: 是否增量处理
            
        Returns:
            是否成功
        """
        if dataset_name not in self.dataset_metadata:
            logger.error(f"数据集不存在: {dataset_name}")
            return False
        
        dataset_path = self.storage_dir / dataset_name
        
        try:
            # 验证数据模式
            if not self._validate_schema(data, dataset_name):
                logger.error("数据模式验证失败")
                return False
            
            # 生成批次ID
            if batch_id is None:
                batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 数据预处理
            processed_data = self._preprocess_data(data, dataset_name)
            
            # 分块处理
            chunks = self._split_into_chunks(processed_data)
            
            # 保存数据块
            chunk_files = []
            for i, chunk in enumerate(chunks):
                chunk_file = self._save_data_chunk(chunk, dataset_path, batch_id, i)
                chunk_files.append(chunk_file)
            
            # 更新元数据
            self._update_dataset_metadata(dataset_name, len(processed_data), chunk_files)
            
            # 构建索引
            if incremental:
                self._build_incremental_index(dataset_name, batch_id, processed_data)
            
            logger.info(f"数据添加成功: {dataset_name}, 批次: {batch_id}, 记录数: {len(processed_data)}")
            return True
            
        except Exception as e:
            logger.error(f"添加数据失败: {e}")
            return False
    
    def _validate_schema(self, data: pd.DataFrame, dataset_name: str) -> bool:
        """验证数据模式"""
        schema = self.dataset_metadata[dataset_name]["schema"]
        
        for column, expected_type in schema.items():
            if column not in data.columns:
                logger.warning(f"缺少列: {column}")
                # 添加缺失列
                if expected_type == "int":
                    data[column] = 0
                elif expected_type == "float":
                    data[column] = 0.0
                elif expected_type == "str":
                    data[column] = ""
                elif expected_type == "bool":
                    data[column] = False
                else:
                    data[column] = None
        
        # 类型转换
        for column, expected_type in schema.items():
            if column in data.columns:
                try:
                    if expected_type == "int":
                        data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0).astype(int)
                    elif expected_type == "float":
                        data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0.0).astype(float)
                    elif expected_type == "bool":
                        data[column] = data[column].astype(bool)
                except Exception as e:
                    logger.warning(f"列类型转换失败 {column}: {e}")
        
        return True
    
    def _preprocess_data(self, data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """数据预处理"""
        processed_data = data.copy()
        
        # 处理时间戳
        if 'timestamp' in processed_data.columns:
            processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
        
        # 处理无穷值和NaN
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        processed_data[numeric_cols] = processed_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 填充缺失值
        for col in processed_data.columns:
            if processed_data[col].dtype in ['int64', 'float64']:
                processed_data[col] = processed_data[col].fillna(0)
            elif processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].fillna('')
        
        # 内存优化
        processed_data = self._optimize_memory(processed_data)
        
        return processed_data
    
    def _optimize_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """优化内存使用"""
        optimized_data = data.copy()
        
        # 优化数值列
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            col_min = optimized_data[col].min()
            col_max = optimized_data[col].max()
            
            # 整数类型优化
            if optimized_data[col].dtype == np.int64:
                if col_min >= 0:
                    if col_max < 255:
                        optimized_data[col] = optimized_data[col].astype(np.uint8)
                    elif col_max < 65535:
                        optimized_data[col] = optimized_data[col].astype(np.uint16)
                    elif col_max < 4294967295:
                        optimized_data[col] = optimized_data[col].astype(np.uint32)
                else:
                    if col_min > -128 and col_max < 127:
                        optimized_data[col] = optimized_data[col].astype(np.int8)
                    elif col_min > -32768 and col_max < 32767:
                        optimized_data[col] = optimized_data[col].astype(np.int16)
                    elif col_min > -2147483648 and col_max < 2147483647:
                        optimized_data[col] = optimized_data[col].astype(np.int32)
            
            # 浮点数类型优化
            elif optimized_data[col].dtype == np.float64:
                if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    optimized_data[col] = optimized_data[col].astype(np.float32)
        
        # 优化分类数据
        for col in optimized_data.select_dtypes(include=['object']).columns:
            num_unique = optimized_data[col].nunique()
            if num_unique / len(optimized_data) < 0.5:  # 如果唯一值比例小于50%
                optimized_data[col] = optimized_data[col].astype('category')
        
        original_memory = data.memory_usage(deep=True).sum() / 1024**2
        optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024**2
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"内存优化: {original_memory:.2f}MB -> {optimized_memory:.2f}MB ({reduction:.1f}% 减少)")
        
        return optimized_data
    
    def _split_into_chunks(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """将数据分块"""
        chunks = []
        total_rows = len(data)
        
        for start in range(0, total_rows, self.chunk_size):
            end = min(start + self.chunk_size, total_rows)
            chunk = data.iloc[start:end].copy()
            chunks.append(chunk)
        
        logger.info(f"数据分块完成: {len(chunks)} 个块, 每块最多 {self.chunk_size} 条记录")
        return chunks
    
    def _save_data_chunk(self, 
                        chunk: pd.DataFrame, 
                        dataset_path: Path, 
                        batch_id: str, 
                        chunk_index: int) -> Dict[str, str]:
        """保存数据块"""
        chunk_files = {}
        
        # CSV格式
        csv_file = dataset_path / "data" / f"{batch_id}_chunk_{chunk_index}.csv"
        chunk.to_csv(csv_file, index=False)
        chunk_files['csv'] = str(csv_file)
        
        # 压缩的pickle格式 (用于快速加载)
        pickle_file = dataset_path / "cache" / f"{batch_id}_chunk_{chunk_index}.pkl"
        if self.compression:
            with open(pickle_file, 'wb') as f:
                compressed_data = zlib.compress(pickle.dumps(chunk))
                f.write(compressed_data)
        else:
            with open(pickle_file, 'wb') as f:
                pickle.dump(chunk, f)
        chunk_files['pickle'] = str(pickle_file)
        
        logger.debug(f"数据块保存完成: {batch_id}_chunk_{chunk_index}")
        return chunk_files
    
    def _update_dataset_metadata(self, 
                               dataset_name: str, 
                               new_records: int,
                               chunk_files: List[Dict[str, str]]):
        """更新数据集元数据"""
        metadata = self.dataset_metadata[dataset_name]
        metadata["total_records"] += new_records
        metadata["last_updated"] = datetime.now().isoformat()
        
        # 计算总大小
        total_size = 0
        for chunk_file in chunk_files:
            for file_path in chunk_file.values():
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        metadata["total_size_mb"] = total_size / 1024 / 1024
        
        # 保存更新后的元数据
        metadata_file = self.storage_dir / dataset_name / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.dataset_metadata[dataset_name] = metadata
    
    def _build_incremental_index(self, 
                               dataset_name: str, 
                               batch_id: str, 
                               data: pd.DataFrame):
        """构建增量索引"""
        dataset_path = self.storage_dir / dataset_name
        index_file = dataset_path / "indices" / f"{batch_id}_index.json"
        
        # 创建时间范围索引
        if 'timestamp' in data.columns:
            time_index = {
                "batch_id": batch_id,
                "min_timestamp": data['timestamp'].min().isoformat(),
                "max_timestamp": data['timestamp'].max().isoformat(),
                "record_count": len(data),
                "anomaly_count": int(data.get('is_anomaly', 0).sum()) if 'is_anomaly' in data.columns else 0
            }
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(time_index, f, indent=2, ensure_ascii=False)
        
        # 更新全局索引
        self._update_global_index(dataset_name)
    
    def _update_global_index(self, dataset_name: str):
        """更新全局索引"""
        dataset_path = self.storage_dir / dataset_name
        indices_dir = dataset_path / "indices"
        
        global_index = {
            "dataset_name": dataset_name,
            "last_updated": datetime.now().isoformat(),
            "batches": [],
            "time_range": {"min": None, "max": None},
            "total_records": 0,
            "total_anomalies": 0
        }
        
        # 收集所有批次索引
        for index_file in indices_dir.glob("*_index.json"):
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    batch_index = json.load(f)
                    global_index["batches"].append(batch_index)
                    
                    # 更新时间范围
                    if batch_index["min_timestamp"]:
                        batch_min = datetime.fromisoformat(batch_index["min_timestamp"])
                        if global_index["time_range"]["min"] is None or batch_min < datetime.fromisoformat(global_index["time_range"]["min"]):
                            global_index["time_range"]["min"] = batch_index["min_timestamp"]
                    
                    if batch_index["max_timestamp"]:
                        batch_max = datetime.fromisoformat(batch_index["max_timestamp"])
                        if global_index["time_range"]["max"] is None or batch_max > datetime.fromisoformat(global_index["time_range"]["max"]):
                            global_index["time_range"]["max"] = batch_index["max_timestamp"]
                    
                    global_index["total_records"] += batch_index["record_count"]
                    global_index["total_anomalies"] += batch_index["anomaly_count"]
                    
            except Exception as e:
                logger.warning(f"加载批次索引失败 {index_file}: {e}")
        
        # 保存全局索引
        global_index_file = dataset_path / "global_index.json"
        with open(global_index_file, 'w', encoding='utf-8') as f:
            json.dump(global_index, f, indent=2, ensure_ascii=False)
    
    def query_dataset(self, 
                     dataset_name: str,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     filters: Optional[Dict[str, Any]] = None,
                     columns: Optional[List[str]] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """
        查询数据集
        
        Args:
            dataset_name: 数据集名称
            start_time: 开始时间
            end_time: 结束时间
            filters: 过滤条件
            columns: 选择的列
            limit: 限制记录数
            
        Returns:
            查询结果DataFrame
        """
        if dataset_name not in self.dataset_metadata:
            logger.error(f"数据集不存在: {dataset_name}")
            return pd.DataFrame()
        
        dataset_path = self.storage_dir / dataset_name
        
        try:
            # 加载全局索引确定需要查询的批次
            relevant_batches = self._find_relevant_batches(dataset_name, start_time, end_time)
            
            if not relevant_batches:
                logger.info("没有找到符合条件的数据批次")
                return pd.DataFrame()
            
            # 并行加载数据
            results = []
            with ThreadPoolExecutor(max_workers=min(len(relevant_batches), 4)) as executor:
                future_to_batch = {
                    executor.submit(self._load_batch_data, dataset_path, batch, columns): batch
                    for batch in relevant_batches
                }
                
                for future in future_to_batch:
                    try:
                        batch_data = future.result()
                        if not batch_data.empty:
                            results.append(batch_data)
                    except Exception as e:
                        logger.warning(f"加载批次数据失败: {e}")
            
            if not results:
                return pd.DataFrame()
            
            # 合并结果
            combined_data = pd.concat(results, ignore_index=True)
            
            # 应用时间过滤
            if start_time and 'timestamp' in combined_data.columns:
                combined_data = combined_data[combined_data['timestamp'] >= start_time]
            if end_time and 'timestamp' in combined_data.columns:
                combined_data = combined_data[combined_data['timestamp'] <= end_time]
            
            # 应用其他过滤条件
            if filters:
                for column, value in filters.items():
                    if column in combined_data.columns:
                        if isinstance(value, (list, tuple)):
                            combined_data = combined_data[combined_data[column].isin(value)]
                        else:
                            combined_data = combined_data[combined_data[column] == value]
            
            # 应用列选择
            if columns:
                available_columns = [col for col in columns if col in combined_data.columns]
                combined_data = combined_data[available_columns]
            
            # 应用限制
            if limit and limit > 0:
                combined_data = combined_data.head(limit)
            
            logger.info(f"查询完成: {len(combined_data)} 条记录")
            return combined_data
            
        except Exception as e:
            logger.error(f"查询数据集失败: {e}")
            return pd.DataFrame()
    
    def _find_relevant_batches(self, 
                             dataset_name: str,
                             start_time: Optional[datetime],
                             end_time: Optional[datetime]) -> List[str]:
        """查找相关批次"""
        dataset_path = self.storage_dir / dataset_name
        global_index_file = dataset_path / "global_index.json"
        
        if not global_index_file.exists():
            # 如果没有全局索引，返回所有批次
            data_dir = dataset_path / "data"
            batches = set()
            for file in data_dir.glob("*_chunk_*.csv"):
                batch_id = file.name.split('_chunk_')[0]
                batches.add(batch_id)
            return list(batches)
        
        with open(global_index_file, 'r', encoding='utf-8') as f:
            global_index = json.load(f)
        
        relevant_batches = []
        
        for batch in global_index["batches"]:
            batch_min = datetime.fromisoformat(batch["min_timestamp"])
            batch_max = datetime.fromisoformat(batch["max_timestamp"])
            
            # 检查时间重叠
            time_overlap = True
            if start_time and batch_max < start_time:
                time_overlap = False
            if end_time and batch_min > end_time:
                time_overlap = False
            
            if time_overlap:
                relevant_batches.append(batch["batch_id"])
        
        return relevant_batches
    
    def _load_batch_data(self, 
                        dataset_path: Path, 
                        batch_id: str,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """加载批次数据"""
        batch_files = list((dataset_path / "data").glob(f"{batch_id}_chunk_*.csv"))
        
        if not batch_files:
            return pd.DataFrame()
        
        # 加载所有块
        chunks = []
        for file in batch_files:
            try:
                chunk_data = pd.read_csv(file, usecols=columns)
                chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"加载数据块失败 {file}: {e}")
        
        if not chunks:
            return pd.DataFrame()
        
        # 合并块
        batch_data = pd.concat(chunks, ignore_index=True)
        return batch_data
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        if dataset_name not in self.dataset_metadata:
            return {}
        
        metadata = self.dataset_metadata[dataset_name].copy()
        
        # 添加文件统计
        dataset_path = self.storage_dir / dataset_name
        data_files = list((dataset_path / "data").glob("*.*"))
        metadata["file_count"] = len(data_files)
        
        # 添加最新批次信息
        global_index_file = dataset_path / "global_index.json"
        if global_index_file.exists():
            with open(global_index_file, 'r', encoding='utf-8') as f:
                global_index = json.load(f)
            metadata["batch_count"] = len(global_index["batches"])
            metadata["global_index"] = global_index
        
        return metadata
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """删除数据集"""
        if dataset_name not in self.dataset_metadata:
            logger.error(f"数据集不存在: {dataset_name}")
            return False
        
        try:
            dataset_path = self.storage_dir / dataset_name
            
            # 递归删除目录
            import shutil
            shutil.rmtree(dataset_path)
            
            # 从内存中移除
            del self.dataset_metadata[dataset_name]
            
            logger.info(f"数据集删除成功: {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除数据集失败: {e}")
            return False
    
    def export_dataset(self, 
                       dataset_name: str, 
                       output_format: str = "csv",
                       output_dir: Optional[str] = None) -> str:
        """
        导出数据集
        
        Args:
            dataset_name: 数据集名称
            output_format: 输出格式 ('csv', 'pickle')
            output_dir: 输出目录
            
        Returns:
            输出文件路径
        """
        if dataset_name not in self.dataset_metadata:
            logger.error(f"数据集不存在: {dataset_name}")
            return ""
        
        if output_dir is None:
            output_dir = self.storage_dir / "exports"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 查询所有数据
            all_data = self.query_dataset(dataset_name)
            
            if all_data.empty:
                logger.warning("数据集为空，无法导出")
                return ""
            
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{dataset_name}_{timestamp}.{output_format}"
            
            # 导出数据
            if output_format == "csv":
                all_data.to_csv(output_file, index=False)
            elif output_format == "pickle":
                with open(output_file, "wb") as f:
                    pickle.dump(all_data, f)
            elif output_format == "parquet":
                try:
                    # 需要 pyarrow 或 fastparquet 作为后端
                    all_data.to_parquet(output_file, index=False)
                except ImportError:
                    logger.error("Parquet导出失败: 缺少pyarrow或fastparquet依赖，请安装后重试")
                    return ""
                except Exception as e:
                    logger.error(f"Parquet导出失败: {e}")
                    return ""
            else:
                logger.error(f"不支持的输出格式: {output_format}")
                return ""
            
            logger.info(f"数据集导出成功: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"导出数据集失败: {e}")
            return ""


# 便捷函数
def create_multi_model_dataset_manager() -> LargeDatasetManager:
    """创建多模型实验数据集管理器"""
    # 定义TSE-Matrix数据模式
    schema = {
        "timestamp": "datetime",
        "is_anomaly": "int",
        "gpu_utilization_gpu": "float",
        "gpu_utilization_memory": "float",
        "gpu_memory_total": "int",
        "gpu_memory_used": "int",
        "gpu_memory_free": "int",
        "gpu_temperature": "float",
        "gpu_power_draw": "float",
        "sys_cpu_percent": "float",
        "sys_memory_percent": "float",
        "sys_disk_usage": "float",
        "train_step": "int",
        "train_loss": "float",
        "train_learning_rate": "float",
        "train_throughput": "float",
        "train_step_time": "float",
        "eval_accuracy": "float",
        "eval_loss": "float",
        "event_anomaly": "int",
        "gpu_memory_utilization_ratio": "float",
        "train_step_rate": "float",
        "train_loss_change_rate": "float",
        "total_anomaly_events": "int"
    }
    
    manager = LargeDatasetManager(
        storage_dir="./multi_model_datasets",
        compression=True,
        chunk_size=5000,
        max_memory_mb=2048
    )
    
    manager.create_dataset(
        dataset_name="multi_model_tse_matrix",
        schema=schema,
        description="多模型实验TSE-Matrix数据集",
        tags=["anomaly_detection", "multi_model", "tse_matrix"]
    )
    
    return manager


if __name__ == "__main__":
    # 测试数据集管理器
    manager = create_multi_model_dataset_manager()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1S'),
        'is_anomaly': np.random.randint(0, 2, 100),
        'gpu_utilization_gpu': np.random.uniform(0, 100, 100),
        'gpu_utilization_memory': np.random.uniform(0, 100, 100),
        'gpu_memory_used': np.random.randint(0, 10000, 100),
        'sys_cpu_percent': np.random.uniform(0, 100, 100),
        'train_loss': np.random.uniform(0, 2, 100),
        'event_anomaly': np.random.randint(0, 2, 100)
    })
    
    # 添加数据
    success = manager.add_data_to_dataset("multi_model_tse_matrix", test_data)
    print(f"数据添加: {'成功' if success else '失败'}")
    
    # 查询数据
    result = manager.query_dataset("multi_model_tse_matrix", limit=10)
    print(f"查询结果: {len(result)} 条记录")
    
    # 获取数据集信息
    info = manager.get_dataset_info("multi_model_tse_matrix")
    print(f"数据集信息: {info}")