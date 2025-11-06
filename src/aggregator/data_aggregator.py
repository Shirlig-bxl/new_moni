"""
数据聚合模块
负责将多源监控数据（GPU、系统、训练日志）进行时间戳对齐和聚合
支持并行处理和内存优化
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import re
import argparse
from pathlib import Path
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

logger = logging.getLogger(__name__)


class DataAggregator:
    """多源数据聚合器"""
    
    def __init__(self, time_granularity: int = 1, use_parallel: bool = True,
                 chunk_size: int = 10000, optimize_memory: bool = True):
        """
        初始化数据聚合器
        
        Args:
            time_granularity: 时间粒度（秒）
            use_parallel: 是否使用并行处理
            chunk_size: 分块处理大小
            optimize_memory: 是否优化内存使用
        """
        self.time_granularity = time_granularity
        self.use_parallel = use_parallel
        self.chunk_size = chunk_size
        self.optimize_memory = optimize_memory
        self.aggregated_data = None
        
        logger.info(f"数据聚合器初始化完成: 时间粒度={time_granularity}s, 并行={use_parallel}, 分块={chunk_size}")
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化DataFrame内存使用
        
        Args:
            df: 输入DataFrame
            
        Returns:
            优化后的DataFrame
        """
        if not self.optimize_memory or df.empty:
            return df
        
        df_optimized = df.copy()
        
        # 优化数值列的数据类型
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            # 整数类型优化
            if df_optimized[col].dtype == np.int64:
                if col_min >= 0:
                    if col_max < 255:
                        df_optimized[col] = df_optimized[col].astype(np.uint8)
                    elif col_max < 65535:
                        df_optimized[col] = df_optimized[col].astype(np.uint16)
                    elif col_max < 4294967295:
                        df_optimized[col] = df_optimized[col].astype(np.uint32)
                else:
                    if col_min > -128 and col_max < 127:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif col_min > -32768 and col_max < 32767:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif col_min > -2147483648 and col_max < 2147483647:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
            
            # 浮点数类型优化
            elif df_optimized[col].dtype == np.float64:
                if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # 优化分类数据
        for col in df_optimized.select_dtypes(include=['object']).columns:
            num_unique = df_optimized[col].nunique()
            if num_unique / len(df_optimized) < 0.5:  # 如果唯一值比例小于50%
                df_optimized[col] = df_optimized[col].astype('category')
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"内存优化: {original_memory:.2f}MB -> {optimized_memory:.2f}MB ({reduction:.1f}% 减少)")
        
        return df_optimized
    
    def _load_data_parallel(self, file_paths: List[str], load_funcs: List[callable]) -> List[pd.DataFrame]:
        """
        并行加载多个数据文件
        
        Args:
            file_paths: 文件路径列表
            load_funcs: 加载函数列表
            
        Returns:
            数据框列表
        """
        if not self.use_parallel or len(file_paths) <= 1:
            # 串行处理
            results = []
            for file_path, load_func in zip(file_paths, load_funcs):
                results.append(load_func(file_path))
            return results
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=min(len(file_paths), mp.cpu_count())) as executor:
            futures = []
            for file_path, load_func in zip(file_paths, load_funcs):
                futures.append(executor.submit(load_func, file_path))
            
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"并行加载数据失败: {e}")
                    results.append(pd.DataFrame())
        
        return results
    
    def load_gpu_metrics(self, gpu_file: str) -> pd.DataFrame:
        """
        加载GPU监控数据
        
        Args:
            gpu_file: GPU监控CSV文件路径
            
        Returns:
            GPU数据DataFrame
        """
        print(f"加载GPU监控数据: {gpu_file}")
        
        try:
            df = pd.read_csv(gpu_file)
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 如果有多个GPU，需要聚合
            if 'gpu_id' in df.columns:
                # 按时间戳聚合多GPU数据
                gpu_agg = df.groupby('timestamp').agg({
                    'utilization_gpu': 'mean',
                    'utilization_memory': 'mean',
                    'memory_total': 'sum',
                    'memory_used': 'sum',
                    'memory_free': 'sum',
                    'memory_percent': 'mean',
                    'temperature': 'mean',
                    'power_draw': 'sum'
                }).reset_index()
                
                # 添加GPU数量信息
                gpu_count = df.groupby('timestamp')['gpu_id'].nunique().reset_index()
                gpu_agg = gpu_agg.merge(gpu_count, on='timestamp')
                gpu_agg.rename(columns={'gpu_id': 'gpu_count'}, inplace=True)
            else:
                gpu_agg = df.copy()
            
            # 重命名列以避免冲突
            gpu_columns = {col: f'gpu_{col}' for col in gpu_agg.columns if col != 'timestamp'}
            gpu_agg.rename(columns=gpu_columns, inplace=True)
            
            print(f"GPU数据加载完成: {len(gpu_agg)} 条记录")
            return gpu_agg
            
        except Exception as e:
            print(f"加载GPU数据失败: {e}")
            return pd.DataFrame()
    
    def load_system_metrics(self, system_file: str) -> pd.DataFrame:
        """
        加载系统监控数据
        
        Args:
            system_file: 系统监控CSV文件路径
            
        Returns:
            系统数据DataFrame
        """
        print(f"加载系统监控数据: {system_file}")
        
        try:
            df = pd.read_csv(system_file)
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 重命名列以避免冲突
            system_columns = {col: f'sys_{col}' for col in df.columns if col != 'timestamp'}
            df.rename(columns=system_columns, inplace=True)
            
            print(f"系统数据加载完成: {len(df)} 条记录")
            return df
            
        except Exception as e:
            print(f"加载系统数据失败: {e}")
            return pd.DataFrame()
    
    def load_training_metrics(self, training_file: str) -> pd.DataFrame:
        """
        加载训练指标数据
        
        Args:
            training_file: 训练指标CSV文件路径
            
        Returns:
            训练数据DataFrame
        """
        print(f"加载训练指标数据: {training_file}")
        
        try:
            df = pd.read_csv(training_file)
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 处理训练指标
            # 对于每个时间戳，可能有多条记录（训练步骤和评估），需要聚合
            
            # 分离训练步骤和评估数据
            train_steps = df[df['metric_type'] == 'training_step'].copy()
            eval_data = df[df['metric_type'] == 'evaluation'].copy()
            
            # 聚合训练步骤数据（按时间戳）
            if not train_steps.empty:
                train_agg = train_steps.groupby('timestamp').agg({
                    'step': 'max',  # 最大步数
                    'loss': 'last',  # 最后的loss值
                    'learning_rate': 'last',  # 最后的学习率
                    'throughput': 'mean',  # 平均吞吐量
                    'step_time': 'mean'  # 平均步时间
                }).reset_index()
                
                # 重命名训练列
                train_columns = {col: f'train_{col}' for col in train_agg.columns if col != 'timestamp'}
                train_agg.rename(columns=train_columns, inplace=True)
            else:
                train_agg = pd.DataFrame()
            
            # 聚合评估数据
            if not eval_data.empty:
                eval_cols = [col for col in eval_data.columns if col.startswith('eval_')]
                if eval_cols:
                    eval_agg = eval_data.groupby('timestamp')[eval_cols].last().reset_index()
                else:
                    eval_agg = pd.DataFrame()
            else:
                eval_agg = pd.DataFrame()
            
            # 聚合事件数据
            event_cols = [col for col in df.columns if col.startswith('event_')]
            if event_cols:
                event_agg = df.groupby('timestamp')[event_cols].max().reset_index()  # 任何事件发生就标记为1
            else:
                event_agg = pd.DataFrame()
            
            # 合并所有训练相关数据
            training_data = pd.DataFrame({'timestamp': df['timestamp'].unique()})
            
            if not train_agg.empty:
                training_data = training_data.merge(train_agg, on='timestamp', how='left')
            
            if not eval_agg.empty:
                training_data = training_data.merge(eval_agg, on='timestamp', how='left')
            
            if not event_agg.empty:
                training_data = training_data.merge(event_agg, on='timestamp', how='left')
            
            print(f"训练数据加载完成: {len(training_data)} 条记录")
            return training_data
            
        except Exception as e:
            print(f"加载训练数据失败: {e}")
            return pd.DataFrame()
    
    def align_timestamps(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        对齐多个数据源的时间戳
        
        Args:
            dataframes: 数据框列表
            
        Returns:
            时间戳对齐后的合并数据框
        """
        print("开始时间戳对齐...")
        
        # 过滤空数据框
        valid_dfs = [df for df in dataframes if not df.empty]
        
        if not valid_dfs:
            print("没有有效的数据框")
            return pd.DataFrame()
        
        # 找到时间范围
        min_time = min(df['timestamp'].min() for df in valid_dfs)
        max_time = max(df['timestamp'].max() for df in valid_dfs)
        
        print(f"时间范围: {min_time} 到 {max_time}")
        
        # 创建统一的时间网格
        time_range = pd.date_range(
            start=min_time.floor(f'{self.time_granularity}S'),
            end=max_time.ceil(f'{self.time_granularity}S'),
            freq=f'{self.time_granularity}S'
        )
        
        # 创建基础时间框架
        aligned_df = pd.DataFrame({'timestamp': time_range})
        
        # 逐个合并数据框
        for i, df in enumerate(valid_dfs):
            print(f"合并数据框 {i+1}/{len(valid_dfs)}")
            
            # 将时间戳舍入到指定粒度
            df_rounded = df.copy()
            df_rounded['timestamp'] = df_rounded['timestamp'].dt.round(f'{self.time_granularity}S')
            
            # 如果有重复时间戳，取平均值（数值列）或最后值（其他列）
            numeric_cols = df_rounded.select_dtypes(include=[np.number]).columns.tolist()
            if 'timestamp' in numeric_cols:
                numeric_cols.remove('timestamp')
            
            other_cols = [col for col in df_rounded.columns if col not in numeric_cols and col != 'timestamp']
            
            agg_dict = {}
            if numeric_cols:
                agg_dict.update({col: 'mean' for col in numeric_cols})
            if other_cols:
                agg_dict.update({col: 'last' for col in other_cols})
            
            if agg_dict:
                df_grouped = df_rounded.groupby('timestamp').agg(agg_dict).reset_index()
            else:
                df_grouped = df_rounded.drop_duplicates('timestamp')
            
            # 合并到主数据框
            aligned_df = aligned_df.merge(df_grouped, on='timestamp', how='left')
        
        print(f"时间戳对齐完成: {len(aligned_df)} 条记录")
        return aligned_df
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失值
        
        Args:
            df: 输入数据框
            
        Returns:
            填充后的数据框
        """
        print("填充缺失值...")
        
        df_filled = df.copy()
        
        # 数值列：前向填充 + 后向填充 + 0填充
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 事件列：填充为0
        event_cols = [col for col in df_filled.columns if col.startswith('event_')]
        for col in event_cols:
            df_filled[col] = df_filled[col].fillna(0)
        
        # 其他列：前向填充
        other_cols = [col for col in df_filled.columns 
                     if col not in numeric_cols and col not in event_cols and col != 'timestamp']
        for col in other_cols:
            df_filled[col] = df_filled[col].fillna(method='ffill').fillna(method='bfill')
        
        print("缺失值填充完成")
        return df_filled
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加派生特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加派生特征后的数据框
        """
        print("添加派生特征...")
        
        df_enhanced = df.copy()
        
        # GPU相关派生特征
        if 'gpu_memory_used' in df_enhanced.columns and 'gpu_memory_total' in df_enhanced.columns:
            df_enhanced['gpu_memory_utilization_ratio'] = (
                df_enhanced['gpu_memory_used'] / df_enhanced['gpu_memory_total'].replace(0, np.nan)
            ).fillna(0)
        
        # 系统负载相关
        if 'sys_cpu_percent' in df_enhanced.columns:
            df_enhanced['sys_cpu_load_level'] = pd.cut(
                df_enhanced['sys_cpu_percent'], 
                bins=[0, 25, 50, 75, 100], 
                labels=['low', 'medium', 'high', 'critical']
            )
        
        # 训练进度相关
        if 'train_step' in df_enhanced.columns:
            df_enhanced['train_step_rate'] = df_enhanced['train_step'].diff() / self.time_granularity
            df_enhanced['train_step_rate'] = df_enhanced['train_step_rate'].fillna(0)
        
        # Loss变化率
        if 'train_loss' in df_enhanced.columns:
            df_enhanced['train_loss_change_rate'] = df_enhanced['train_loss'].pct_change().fillna(0)
            df_enhanced['train_loss_moving_avg'] = df_enhanced['train_loss'].rolling(window=10, min_periods=1).mean()
        
        # 异常指标聚合
        event_cols = [col for col in df_enhanced.columns if col.startswith('event_')]
        if event_cols:
            df_enhanced['total_anomaly_events'] = df_enhanced[event_cols].sum(axis=1)
        
        # 时间特征
        df_enhanced['hour'] = df_enhanced['timestamp'].dt.hour
        df_enhanced['minute'] = df_enhanced['timestamp'].dt.minute
        df_enhanced['elapsed_seconds'] = (
            df_enhanced['timestamp'] - df_enhanced['timestamp'].min()
        ).dt.total_seconds()
        
        print("派生特征添加完成")
        return df_enhanced
    
    def aggregate_data(self, gpu_file: str, system_file: str, training_file: str) -> pd.DataFrame:
        """
        聚合多源数据
        
        Args:
            gpu_file: GPU监控文件
            system_file: 系统监控文件
            training_file: 训练指标文件
            
        Returns:
            聚合后的数据框
        """
        print("开始数据聚合...")
        
        # 加载各数据源
        dataframes = []
        
        if gpu_file and os.path.exists(gpu_file):
            gpu_df = self.load_gpu_metrics(gpu_file)
            if not gpu_df.empty:
                dataframes.append(gpu_df)
        
        if os.path.exists(system_file):
            system_df = self.load_system_metrics(system_file)
            if not system_df.empty:
                dataframes.append(system_df)
        
        if os.path.exists(training_file):
            training_df = self.load_training_metrics(training_file)
            if not training_df.empty:
                dataframes.append(training_df)
        
        if not dataframes:
            print("没有找到有效的数据文件")
            return pd.DataFrame()
        
        # 时间戳对齐
        aligned_df = self.align_timestamps(dataframes)
        
        if aligned_df.empty:
            print("时间戳对齐失败")
            return pd.DataFrame()
        
        # 填充缺失值
        filled_df = self.fill_missing_values(aligned_df)
        
        # 添加派生特征
        enhanced_df = self.add_derived_features(filled_df)
        
        self.aggregated_data = enhanced_df
        
        print(f"数据聚合完成: {len(enhanced_df)} 条记录, {len(enhanced_df.columns)} 个特征")
        return enhanced_df
    
    def save_aggregated_data(self, output_file: str):
        """
        保存聚合后的数据
        
        Args:
            output_file: 输出文件路径
        """
        if self.aggregated_data is None or self.aggregated_data.empty:
            print("没有聚合数据可保存")
            return
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存为CSV
        self.aggregated_data.to_csv(output_file, index=False)
        print(f"聚合数据已保存到: {output_file}")
        
        # 保存数据摘要
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("数据聚合摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"记录数量: {len(self.aggregated_data)}\n")
            f.write(f"特征数量: {len(self.aggregated_data.columns)}\n")
            f.write(f"时间范围: {self.aggregated_data['timestamp'].min()} 到 {self.aggregated_data['timestamp'].max()}\n")
            f.write(f"时间粒度: {self.time_granularity}秒\n\n")
            
            f.write("特征列表:\n")
            for i, col in enumerate(self.aggregated_data.columns, 1):
                f.write(f"{i:3d}. {col}\n")
            
            f.write("\n数据统计:\n")
            f.write(str(self.aggregated_data.describe()))
        
        print(f"数据摘要已保存到: {summary_file}")
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Returns:
            数据质量报告字典
        """
        if self.aggregated_data is None or self.aggregated_data.empty:
            return {}
        
        df = self.aggregated_data
        
        report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration_seconds': (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'anomaly_events': {}
        }
        
        # 异常事件统计
        event_cols = [col for col in df.columns if col.startswith('event_')]
        for col in event_cols:
            if col in df.columns:
                report['anomaly_events'][col] = int(df[col].sum())
        
        # 数据完整性
        completeness = (1 - df.isnull().sum() / len(df)) * 100
        report['completeness'] = completeness.to_dict()
        
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多源数据聚合工具")
    parser.add_argument("--gpu-file", required=True, help="GPU监控CSV文件")
    parser.add_argument("--system-file", required=True, help="系统监控CSV文件")
    parser.add_argument("--training-file", required=True, help="训练指标CSV文件")
    parser.add_argument("--output", "-o", required=True, help="输出聚合数据文件")
    parser.add_argument("--time-granularity", type=int, default=1, help="时间粒度（秒）")
    
    args = parser.parse_args()
    
    # 创建聚合器
    aggregator = DataAggregator(time_granularity=args.time_granularity)
    
    # 聚合数据
    result = aggregator.aggregate_data(
        gpu_file=args.gpu_file,
        system_file=args.system_file,
        training_file=args.training_file
    )
    
    if not result.empty:
        # 保存结果
        aggregator.save_aggregated_data(args.output)
        
        # 打印质量报告
        quality_report = aggregator.get_data_quality_report()
        print("\n数据质量报告:")
        print(f"总记录数: {quality_report['total_records']}")
        print(f"总特征数: {quality_report['total_features']}")
        print(f"时间跨度: {quality_report['time_range']['duration_seconds']:.1f}秒")
        
        if quality_report['anomaly_events']:
            print("异常事件统计:")
            for event, count in quality_report['anomaly_events'].items():
                if count > 0:
                    print(f"  {event}: {count}")
    else:
        print("数据聚合失败")


if __name__ == "__main__":
    main()