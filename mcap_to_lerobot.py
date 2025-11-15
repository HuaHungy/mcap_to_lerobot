#!/usr/bin/env python3
"""
MCAP 到 LeRobot v2.1 标准格式转换器

📹 支持360p和720p分辨率 - 可通过 --resolution 参数选择（默认720p: 1280x720）
在流式写入视频时实时缩放到目标分辨率，减少存储空间和训练加载时间

功能特性:
1. 30Hz 基准时间戳同步（基于ROS header.stamp）
2. Linux 服务器优化 - 针对多核CPU和高效内存管理
3. 内存优化处理，支持大型MCAP文件
4. 分批处理数据，避免内存溢出
5. 基于 mcap_topic.xlsx 具身双臂topic 配置
6. LeRobot v2.1 标准格式输出（MP4视频 + 分层目录）
7. 数据质量评估报告
8. 主臂、从臂和夹爪曲线图绘制
9. 视频实时缩放到目标分辨率（支持360p/720p，通过--resolution参数选择）

LeRobot v2.1 标准特性:
- MP4 视频格式（替代JPG图像）
- 分层目录结构（meta/data/videos）
- 每集一个Parquet文件（episode_XXXXXX.parquet）
- 每集每相机一个MP4文件
- 完整的元数据（info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl）

使用方法:
python mcap_to_lerobot.py \
  --input /mnt/nas/synnas/docker2/外部数据/外来睿尔曼2000条/GroceryStore_Restrocking_Fallen \
  --output /home/kemove/mcap_to_lerobot/test1 \
  --ffmpeg-threads 32 \
  --ffmpeg-cpu-used 8 \
  --resolution 720p \
  --no-plot
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gc
import os
import shutil
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import time
from tqdm import tqdm

# MCAP 相关导入
try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory as mcap_ros2_decoder
except ImportError:
    print("请安装 mcap 相关依赖:")
    print("pip install mcap mcap-ros2-support")
    sys.exit(1)

# Linux 服务器优化配置
def optimize_for_linux_server():
    """针对 Linux 服务器进行优化配置"""
    # 设置 NumPy 使用优化的 BLAS（根据CPU核心数动态调整）
    cpu_count = psutil.cpu_count()
    # 对于多核系统，可以适当增加线程数以提高性能
    # 但为了避免过度竞争，限制为CPU核心数的一半
    optimal_threads = max(1, cpu_count // 2)
    
    os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    
    # 设置 pandas 使用更高效的内存模式
    pd.set_option('mode.chained_assignment', None)
    pd.set_option('compute.use_bottleneck', True)
    pd.set_option('compute.use_numexpr', True)
    
    # 设置 matplotlib 后端（Linux 服务器通常没有显示）
    plt.switch_backend('Agg')  # 使用非交互式后端
    
    # 设置 OpenCV 优化（根据CPU核心数调整）
    cv2.setNumThreads(optimal_threads)
    
    # 设置垃圾回收优化
    gc.set_threshold(700, 10, 10)  # 更积极的垃圾回收
    
    print("✅ Linux 服务器优化配置已启用")
    print(f"   系统内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   CPU 核心数: {cpu_count}")
    print(f"   使用线程数: {optimal_threads}")
    print(f"   可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")

# 初始化 Linux 服务器优化
optimize_for_linux_server()

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 主要配置常量
TARGET_FREQUENCY = 30  # Hz - 统一目标采样频率
MAX_PROCESSING_DURATION_SECONDS = 300  # 最大处理时长（5分钟）
BATCH_SIZE = 500  # 批处理大小（100秒数据优化：减小batch size）
CHUNK_SIZE = 500  # 分块大小（100秒数据优化：减小chunk size）

# 预处理降采样配置（减少插值误差）
USE_PREPROCESSING_DOWNSAMPLE = True  # 启用预处理降采样
PREPROCESSING_CONFIG = {
    'leader_target_freq': 60,    # 主臂预处理目标频率 (62Hz → 60Hz)
    'follower_target_freq': 180,  # 从臂预处理目标频率 (200Hz → 180Hz)
    'use_antialiasing': True      # 使用抗混叠滤波器
}

# 同步容差（秒）- 基于各模态频率，控制时间对齐严格度（优化版本）
# 相机30Hz：帧间隔33.3ms，采用半帧容差确保同步质量
CAMERA_MAX_DRIFT_S = 1.0 / (2 * 30.0)  # ~16.7ms (半帧容差)
# 主臂关节62Hz：周期16.13ms，采用更严格阈值
LEADER_JOINT_MAX_DRIFT_S = 0.005  # ~5ms (更严格)
# 从臂关节200Hz：周期5ms，采用更严格阈值
FOLLOWER_JOINT_MAX_DRIFT_S = 0.002  # ~2ms (更严格)
# 通用容差（更严格）
GENERIC_MAX_DRIFT_S = 0.015  # 15ms (更严格)

@dataclass
class TopicConfig:
    """Topic配置信息"""
    name: str
    datatype: str
    frequency: float
    description: str
    detection_dimension: str
    message_count: int = 0

@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    topic_name: str
    total_messages: int
    valid_messages: int
    missing_rate: float
    frequency_deviation: float
    data_continuity_score: float
    field_completeness: Dict[str, float]
    timestamp_gaps: List[float]
    quality_score: float

@dataclass
class ConversionReport:
    """转换报告"""
    total_topics: int
    converted_frames: int
    processing_time: float
    quality_metrics: List[DataQualityMetrics]
    overall_quality_score: float
    conversion_issues: List[str]
    timestamp: str

class MCAPToLeRobotV21StandardConverter:
    """MCAP 到 LeRobot v2.1 标准格式转换器 - Linux 服务器优化版本"""

    def __init__(self, mcap_file_path: str, output_dir: str, 
                 max_duration: int = 0, resolution: str = '720p'):
        """初始化转换器
        
        Args:
            resolution: 目标分辨率，支持 '360p' (640x360) 或 '720p' (1280x720)
        """
        # 归一化路径：兼容 Windows 风格反斜杠，统一为 POSIX
        self.mcap_file_path = Path(str(mcap_file_path).replace('\\', '/')).resolve()
        # 同时提供 self.input_path 别名，便于外部调用保持一致
        self.input_path = self.mcap_file_path
        self.output_dir = Path(str(output_dir).replace('\\', '/')).resolve()
        self.max_duration = max_duration
        # 初始化日志输出
        # print(f"[Init] Input: {self.input_path.as_posix()}")
        # print(f"[Init] Output: {self.output_dir.as_posix()}")
        
        
        # 设置目标分辨率
        if resolution == '720p':
            self.target_width = 1280
            self.target_height = 720
            self.resolution_name = '720p'
        else:  # 默认720p，但保留360p支持
            self.target_width = 640
            self.target_height = 360
            self.resolution_name = '360p'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建 v2.1 标准目录结构
        self.meta_dir = self.output_dir / "meta"
        self.data_dir = self.output_dir / "data"
        self.videos_dir = self.output_dir / "videos"
        
        # V2.1 不需要 episodes 子目录，直接在 meta 下
        for dir_path in [self.meta_dir, self.data_dir, self.videos_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 数据存储 - 使用更节省内存的方式
        self.raw_data = defaultdict(list)
        self.timestamps = defaultdict(list)
        self.synchronized_data = {}
        self.target_timestamps = []
        self.topic_configs = {}
        self.data_chunks = []  # 分批存储数据

        # 质量评估
        self.quality_metrics = []
        self.conversion_issues = []

        # M 芯片性能监控
        self.performance_stats = {
            'memory_usage': [],
            'processing_times': [],
            'gc_counts': []
        }

        print(f"开始转换 [{self.resolution_name},{TARGET_FREQUENCY}Hz]: {self.mcap_file_path.name}")
        
        # 加载 Topic 配置（不再支持Excel，直接从MCAP自动探测）
        self.discover_topic_configs_from_mcap()

    # Excel 加载功能已移除

    def discover_topic_configs_from_mcap(self):
        """从MCAP自动探测Topic配置（不解码消息，显著加速，并提供进度条）"""
        topic_stats = {}

        try:
            file_size = self.mcap_file_path.stat().st_size
            with self.mcap_file_path.open("rb") as f:
                reader = make_reader(f)
                pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="扫描MCAP", mininterval=0.5, leave=False)
                last_pos = f.tell()

                for schema, channel, message in reader.iter_messages():
                    topic = channel.topic
                    t_ns = message.log_time  # 纳秒时间戳
                    dtype = schema.name if schema and hasattr(schema, "name") else "unknown"

                    s = topic_stats.get(topic)
                    if s is None:
                        topic_stats[topic] = {
                            "count": 1,
                            "first_ns": t_ns,
                            "last_ns": t_ns,
                            "datatype": dtype,
                        }
                    else:
                        s["count"] += 1
                        if t_ns < s["first_ns"]:
                            s["first_ns"] = t_ns
                        if t_ns > s["last_ns"]:
                            s["last_ns"] = t_ns

                    # 用已读取的字节位置更新进度条
                    pos = f.tell()
                    if pos > last_pos:
                        pbar.update(pos - last_pos)
                        last_pos = pos

                pbar.close()
        except KeyboardInterrupt:
            print("\n⚠️ 扫描被中断，使用当前统计结果继续。")
        except Exception as e:
            raise RuntimeError(f"从MCAP探测Topic失败: {e}")

        # 组装 TopicConfig 字典（与全局使用保持一致）
        self.topic_configs = {}
        for name, s in topic_stats.items():
            duration_s = max((s["last_ns"] - s["first_ns"]) / 1e9, 1e-6)
            freq = (s["count"] / duration_s) if duration_s > 0 else 0.0
            self.topic_configs[name] = TopicConfig(
                name=name,
                datatype=s["datatype"],
                frequency=freq,
                description="auto from mcap",
                detection_dimension="auto",
                message_count=s["count"],
            )

        # print(f"检测到 {len(self.topic_configs)} 个topic")
        # 打印部分示例
        for t in list(self.topic_configs.keys())[:10]:
            tc = self.topic_configs[t]
            # print(f"  - {tc.name}: {tc.frequency:.2f} Hz, count={tc.message_count}, type={tc.datatype}")

    def load_topic_configs_from_mcap(self):
        """从 MCAP 自动探测 Topic，估算频率与消息统计（快速扫描，不解码）"""
        print("\n扫描MCAP(简化)...")
        topic_stats = {}
        try:
            total_size = self.mcap_file_path.stat().st_size
            with self.mcap_file_path.open("rb") as f:
                reader = make_reader(f)
                pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="扫描MCAP(简化)", leave=False)
                last_pos = f.tell()
                for schema, channel, message in reader.iter_messages():
                    topic = channel.topic
                    t_ns = message.log_time  # 纳秒时间戳
                    s = topic_stats.get(topic)
                    if s is None:
                        topic_stats[topic] = {
                            "count": 1,
                            "first_ns": t_ns,
                            "last_ns": t_ns,
                            "datatype": (schema.name if schema and hasattr(schema, "name") else "unknown"),
                        }
                    else:
                        s["count"] += 1
                        if t_ns < s["first_ns"]:
                            s["first_ns"] = t_ns
                        if t_ns > s["last_ns"]:
                            s["last_ns"] = t_ns
                    # 更新进度条
                    pos = f.tell()
                    if pos > last_pos:
                        pbar.update(pos - last_pos)
                        last_pos = pos
                pbar.close()
        except Exception as e:
            raise RuntimeError(f"从MCAP探测Topic失败: {e}")

        # 组装 TopicConfig 字典（与全局使用保持一致）
        self.topic_configs = {}
        for name, s in topic_stats.items():
            duration_s = max((s["last_ns"] - s["first_ns"]) / 1e9, 1e-6)
            freq = (s["count"] / duration_s) if duration_s > 0 else 0.0
            self.topic_configs[name] = TopicConfig(
                name=name,
                datatype=s["datatype"],
                frequency=freq,
                description="auto from mcap",
                detection_dimension="auto",
                message_count=s["count"],
            )

        print(f"检测到 {len(self.topic_configs)} 个Topic")
        for t in list(self.topic_configs.keys())[:10]:
            tc = self.topic_configs[t]
            print(f"  - {tc.name}: {tc.frequency:.2f} Hz, count={tc.message_count}, type={tc.datatype}")

    def extract_timestamp_from_header(self, msg):
        """从ROS消息header中提取时间戳 - 优先使用header.stamp"""
        try:
            # 优先使用header.stamp（传感器采集时间）
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
                # 确保时间戳有效（大于0）
                if timestamp > 0:
                    return timestamp
            # 回退到直接stamp字段
            elif hasattr(msg, 'stamp'):
                stamp = msg.stamp
                timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
                if timestamp > 0:
                    return timestamp
            return None
        except Exception:
            return None

    def analyze_mcap_file_for_duration(self):
        """智能分析MCAP文件，确定处理时长和相关信息"""
        print("\n智能分析MCAP文件...")
        
        start_time = None
        end_time = None
        topic_counts = defaultdict(int)
        topic_timestamps = {}  # 记录每个topic的时间戳范围
        
        try:
            with open(self.mcap_file_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[mcap_ros2_decoder()])
                
                message_count = 0
                for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                    topic_name = channel.topic
                    message_count += 1
                    
                    if topic_name not in self.topic_configs:
                        continue
                    
                    # 优先使用header.stamp（传感器采集时间），回退到log_time
                    timestamp = self.extract_timestamp_from_header(ros_msg)
                    if timestamp is None:
                        timestamp = message.log_time * 1e-9
                    
                    # 记录时间范围
                    if start_time is None:
                        start_time = timestamp
                        # 设置全局起始时间戳，确保后续处理使用相同的基准
                        self.start_timestamp = timestamp
                    end_time = timestamp
                    
                    topic_counts[topic_name] += 1
                    
                    # 记录每个topic的时间戳范围
                    if topic_name not in topic_timestamps:
                        topic_timestamps[topic_name] = {'min': timestamp, 'max': timestamp}
                    else:
                        topic_timestamps[topic_name]['min'] = min(topic_timestamps[topic_name]['min'], timestamp)
                        topic_timestamps[topic_name]['max'] = max(topic_timestamps[topic_name]['max'], timestamp)
                    
                    # 每处理10000条消息显示进度
                    if message_count % 10000 == 0:
                        print(f"  已处理 {message_count} 条消息...")
                    
                    # 如果超过最大处理时长，停止分析
                    if self.max_duration > 0 and timestamp - start_time > self.max_duration:
                        break
                        
        except Exception as e:
            raise RuntimeError(f"分析MCAP文件失败: {e}")
        
        if start_time is None or end_time is None:
            raise ValueError("没有找到有效的时间戳数据")
        
        # 确保时间范围正确
        if end_time < start_time:
            start_time, end_time = end_time, start_time
            
        duration = end_time - start_time
        
        # 分析topic数据覆盖情况
        active_topics = 0
        for topic_name, ts_range in topic_timestamps.items():
            topic_duration = ts_range['max'] - ts_range['min']
            coverage = topic_duration / duration if duration > 0 else 0
            if coverage > 0.8:  # 覆盖80%以上时间
                active_topics += 1
        
        print(f"  文件时长: {duration:.2f} 秒")
        print(f"  时间范围: {start_time:.3f}s - {end_time:.3f}s")
        print(f"  相对时间: 0.000s - {duration:.3f}s")
        print(f"  总消息数: {message_count}")
        print(f"  相关topics: {len(topic_counts)}")
        print(f"  活跃topics: {active_topics} (覆盖>80%时间)")
        
        # 如果设置了最大处理时长，限制实际处理时长
        if self.max_duration > 0 and duration > self.max_duration:
            duration = self.max_duration
            end_time = start_time + duration
            print(f"  限制处理时长: {duration:.2f} 秒")
        
        return start_time, end_time, duration, topic_counts

    def _determine_batch_strategy(self, duration):
        """根据数据时长智能选择批次处理策略"""
        import psutil
        
        # 获取系统内存信息
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # 根据时长选择策略
        if duration <= 30:  # 30秒以内
            strategy = {
                'strategy_name': '单批次处理',
                'num_batches': 1,
                'batch_duration': duration,
                'memory_optimized': False,
                'description': '短数据，直接处理'
            }
        elif duration <= 60:  # 1分钟以内
            strategy = {
                'strategy_name': '小批次处理',
                'num_batches': 2,
                'batch_duration': duration / 2,
                'memory_optimized': True,
                'description': '中等数据，2批次处理'
            }
        elif duration <= 120:  # 2分钟以内
            # 100秒数据内存优化策略
            if available_memory_gb > 6:
                strategy = {
                    'strategy_name': '内存优化大批次处理',
                    'num_batches': 5,
                    'batch_duration': duration / 5,
                    'memory_optimized': True,
                    'description': '100秒数据，5批次处理（每批20秒）- 内存优化'
                }
            elif available_memory_gb > 3:
                strategy = {
                    'strategy_name': '内存优化中批次处理',
                    'num_batches': 8,
                    'batch_duration': duration / 8,
                    'memory_optimized': True,
                    'description': '100秒数据，8批次处理（每批12.5秒）- 内存优化'
                }
            else:
                strategy = {
                    'strategy_name': '超内存优化批次处理',
                    'num_batches': 12,
                    'batch_duration': duration / 12,
                    'memory_optimized': True,
                    'description': '100秒数据，12批次处理（每批8.3秒）- 超内存优化'
                }
        elif duration <= 300:  # 5分钟以内
            # 根据可用内存动态调整
            if available_memory_gb > 8:
                strategy = {
                    'strategy_name': '大内存批次处理',
                    'num_batches': 6,
                    'batch_duration': duration / 6,
                    'memory_optimized': True,
                    'description': '长数据，大内存，6批次处理'
                }
            else:
                strategy = {
                    'strategy_name': '内存优化批次处理',
                    'num_batches': 10,
                    'batch_duration': duration / 10,
                    'memory_optimized': True,
                    'description': '长数据，内存受限，10批次处理'
                }
        else:  # 超过5分钟
            strategy = {
                'strategy_name': '超长数据批次处理',
                'num_batches': max(15, int(duration / 20)),  # 每批最多20秒
                'batch_duration': min(20.0, duration / max(15, int(duration / 20))),
                'memory_optimized': True,
                'description': '超长数据，高频率批次处理'
            }
        
        # 确保批次时长合理（5-60秒之间）
        if strategy['batch_duration'] < 5:
            strategy['batch_duration'] = 5
            strategy['num_batches'] = max(1, int(duration / 5))
        elif strategy['batch_duration'] > 60:
            strategy['batch_duration'] = 60
            strategy['num_batches'] = max(1, int(duration / 60))
        
        # 添加内存使用预估和自适应调整
        # 100秒数据内存优化：保守的内存预估
        base_memory_per_10s = 0.03  # 每10秒约0.03GB（内存优化模式）
        estimated_memory_per_batch = min(1.5, duration / strategy['num_batches'] * base_memory_per_10s)
        
        # 根据可用内存调整批次大小 - 100秒数据内存优化
        if available_memory_gb < 2:  # 内存严重不足时
            # 内存严重不足时，大幅增加批次数
            strategy['num_batches'] = min(strategy['num_batches'] * 3, 20)  # 最多20批次
            strategy['batch_duration'] = duration / strategy['num_batches']
            strategy['memory_optimized'] = True
            strategy['strategy_name'] += " (内存严重不足)"
        elif available_memory_gb < 4:  # 内存不足时
            # 内存不足时，增加批次数
            strategy['num_batches'] = min(strategy['num_batches'] * 2, 15)  # 最多15批次
            strategy['batch_duration'] = duration / strategy['num_batches']
            strategy['memory_optimized'] = True
            strategy['strategy_name'] += " (内存不足优化)"
        elif available_memory_gb < 6 and duration > 80:  # 中等内存，长数据
            # 中等内存时，适度增加批次数
            strategy['num_batches'] = min(int(strategy['num_batches'] * 1.5), 10)
            strategy['batch_duration'] = duration / strategy['num_batches']
            strategy['memory_optimized'] = True
            strategy['strategy_name'] += " (中等内存优化)"
        
        strategy['estimated_memory_per_batch'] = f"{estimated_memory_per_batch:.1f}GB"
        strategy['total_estimated_memory'] = f"{estimated_memory_per_batch * strategy['num_batches']:.1f}GB"
        strategy['available_memory'] = f"{available_memory_gb:.1f}GB"
        
        return strategy

    def _read_mcap_data_once(self, start_time, end_time):
        """一次性读取MCAP文件中的所有数据（优化：避免重复读取）"""
        print(f"\n一次性读取MCAP文件数据...")
        
        all_messages = defaultdict(list)
        all_timestamps = defaultdict(list)
        
        # 使用set进行O(1)查找
        valid_topics = set(self.topic_configs.keys())
        
        message_count = 0
        filtered_count = 0
        
        try:
            with open(self.mcap_file_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[mcap_ros2_decoder()])
                
                for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                    message_count += 1
                    topic_name = channel.topic
                    
                    # 快速过滤：O(1)查找
                    if topic_name not in valid_topics:
                        continue
                    
                    # 优先使用header.stamp（传感器采集时间），回退到log_time
                    timestamp = self.extract_timestamp_from_header(ros_msg)
                    if timestamp is None:
                        timestamp = message.log_time * 1e-9
                    
                    # 初始化起始时间戳（第一个有效消息的时间戳）
                    if not hasattr(self, 'start_timestamp'):
                        self.start_timestamp = timestamp
                    
                    # 将绝对时间戳转换为相对时间戳
                    relative_timestamp = timestamp - self.start_timestamp
                    
                    # 检查是否在时间范围内
                    # start_time和end_time是绝对时间戳，需要转换为相对时间戳
                    relative_start_time = start_time - self.start_timestamp
                    relative_end_time = end_time - self.start_timestamp
                    
                    if relative_timestamp < relative_start_time:
                        continue
                    if relative_timestamp > relative_end_time:
                        # 如果超过了结束时间，可以提前退出（如果数据是按时间排序的）
                        # 但为了安全，我们继续读取到文件末尾
                        pass
                    
                    # 存储消息和时间戳
                    all_messages[topic_name].append(ros_msg)
                    all_timestamps[topic_name].append(relative_timestamp)
                    filtered_count += 1
                    
                    # 每处理100000条消息显示进度
                    if message_count % 100000 == 0:
                        print(f"  已处理 {message_count} 条消息，过滤后 {filtered_count} 条...")
        
        except Exception as e:
            raise RuntimeError(f"读取MCAP文件失败: {e}")
        
        print(f"  读取完成: 处理了 {message_count} 条消息，过滤后 {filtered_count} 条")
        print(f"  有效topics: {len(all_messages)}")
        
        return all_messages, all_timestamps
    
    def process_mcap_data_in_batches(self, start_time, end_time):
        """智能分批处理MCAP数据 - 优化版本：根据文件大小选择策略"""
        print(f"\n开始智能分批处理MCAP数据...")
        
        # 计算批处理参数
        total_duration = end_time - start_time
        
        # 如果设置了最大处理时长，使用它；否则使用总时长
        if self.max_duration > 0:
            processing_duration = min(total_duration, self.max_duration)
        else:
            processing_duration = total_duration
        
        # 检查文件大小和可用内存
        file_size_gb = self.mcap_file_path.stat().st_size / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"  文件大小: {file_size_gb:.2f} GB")
        print(f"  可用内存: {available_memory_gb:.2f} GB")
        
        # ⭐ 智能策略：如果文件太大，使用分批读取而不是一次性加载
        # 阈值：文件大小超过可用内存的50%时，使用分批读取
        if file_size_gb > available_memory_gb * 0.5:
            print(f"  ⚠️  文件较大，使用分批读取策略（避免内存溢出）")
            return self._process_in_batches_with_file_reading(start_time, end_time, processing_duration)
        else:
            print(f"  ✅ 文件较小，使用一次性读取策略（更快）")
            return self._process_in_batches_with_memory(start_time, end_time, processing_duration)
    
    def _process_in_batches_with_memory(self, start_time, end_time, processing_duration):
        """使用内存缓存的分批处理（适用于小文件）"""
        # 智能批次策略选择
        batch_strategy = self._determine_batch_strategy(processing_duration)
        num_batches = batch_strategy['num_batches']
        actual_batch_duration = processing_duration / num_batches
        
        print(f"  总时长: {processing_duration:.2f} 秒")
        print(f"  批次策略: {batch_strategy['strategy_name']}")
        print(f"  批次数: {num_batches}")
        print(f"  每批时长: {actual_batch_duration:.2f} 秒")
        
        # ⭐ 优化：一次性读取所有数据（只读取一次文件）
        all_messages, all_timestamps = self._read_mcap_data_once(start_time, end_time)
        
        all_synchronized_data = {}
        
        for batch_idx in range(num_batches):
            batch_start_time = start_time + batch_idx * actual_batch_duration
            batch_end_time = min(start_time + (batch_idx + 1) * actual_batch_duration, end_time)
            
            # 转换为相对时间戳（因为all_timestamps中存储的是相对时间戳）
            relative_batch_start = batch_start_time - self.start_timestamp
            relative_batch_end = batch_end_time - self.start_timestamp
            
            print(f"\n处理批次 {batch_idx + 1}/{num_batches} ({relative_batch_start:.2f}s - {relative_batch_end:.2f}s)")
            
            # ⭐ 优化：从内存中处理批次数据，而不是重新读取文件
            batch_data = self._process_single_batch_from_memory(
                all_messages, all_timestamps, relative_batch_start, relative_batch_end
            )
            
            # 合并批次数据
            for topic_name, data in batch_data.items():
                if topic_name not in all_synchronized_data:
                    all_synchronized_data[topic_name] = []
                all_synchronized_data[topic_name].extend(data)
            
            # 优化垃圾回收：减少频率
            if batch_idx % 5 == 0:  # 每5个批次才回收一次
                gc.collect()
            
            # 监控内存使用
            memory_usage = psutil.Process().memory_info().rss / (1024**3)
            self.performance_stats['memory_usage'].append(memory_usage)
            
            
            # 更积极地清理内存
            if memory_usage > 2.0:  # 超过2GB就清理
                print(f"    内存使用较高，执行额外清理...")
                gc.collect()
        
        self.synchronized_data = all_synchronized_data
        print(f"\n所有批次处理完成，共处理 {len(all_synchronized_data)} 个topics")
        
        # 设置实际处理时长用于质量评估
        self.actual_processing_duration = end_time - start_time
    
    def _process_in_batches_with_file_reading(self, start_time, end_time, processing_duration):
        """流式处理：只读取一次文件，按批次处理并立即释放内存（适用于大文件）"""
        # 智能批次策略选择
        batch_strategy = self._determine_batch_strategy(processing_duration)
        num_batches = batch_strategy['num_batches']
        actual_batch_duration = processing_duration / num_batches
        
        print(f"  总时长: {processing_duration:.2f} 秒")
        print(f"  批次策略: {batch_strategy['strategy_name']}")
        print(f"  批次数: {num_batches}")
        print(f"  每批时长: {actual_batch_duration:.2f} 秒")
        print(f"  ✅ 使用流式处理策略（只读取一次文件，按批次处理）")
        
        # 计算各批次的时间范围
        batch_ranges = []
        for batch_idx in range(num_batches):
            batch_start_time = start_time + batch_idx * actual_batch_duration
            batch_end_time = min(start_time + (batch_idx + 1) * actual_batch_duration, end_time)
            batch_ranges.append((batch_start_time, batch_end_time))
        
        all_synchronized_data = {}
        
        # ⭐ 流式处理：只读取一次文件，按批次收集数据
        print(f"\n开始流式读取文件（只读取一次）...")
        
        # 为每个批次准备数据容器
        batch_messages = [defaultdict(list) for _ in range(num_batches)]
        batch_timestamps = [defaultdict(list) for _ in range(num_batches)]
        
        # 使用set进行O(1)查找
        valid_topics = set(self.topic_configs.keys())
        
        message_count = 0
        filtered_count = 0
        
        try:
            with open(self.mcap_file_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[mcap_ros2_decoder()])
                
                for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                    message_count += 1
                    topic_name = channel.topic
                    
                    # 快速过滤：O(1)查找
                    if topic_name not in valid_topics:
                        continue
                    
                    # 优先使用header.stamp（传感器采集时间），回退到log_time
                    timestamp = self.extract_timestamp_from_header(ros_msg)
                    if timestamp is None:
                        timestamp = message.log_time * 1e-9
                    
                    # 初始化起始时间戳
                    if not hasattr(self, 'start_timestamp'):
                        self.start_timestamp = timestamp
                    
                    # 转换为相对时间戳
                    relative_timestamp = timestamp - self.start_timestamp
                    
                    # 检查时间范围
                    relative_start_time = start_time - self.start_timestamp
                    relative_end_time = end_time - self.start_timestamp
                    
                    if relative_timestamp < relative_start_time:
                        continue
                    if relative_timestamp > relative_end_time:
                        break  # 超过范围，可以提前退出
                    
                    # ⭐ 优化：使用二分查找确定批次（O(log n)而不是O(n)）
                    # 计算相对时间戳在哪个批次
                    relative_timestamp_scaled = relative_timestamp - relative_start_time
                    batch_idx = int(relative_timestamp_scaled / actual_batch_duration)
                    batch_idx = max(0, min(batch_idx, num_batches - 1))  # 确保在有效范围内
                    
                    # 验证是否在批次范围内
                    relative_batch_start = batch_ranges[batch_idx][0] - self.start_timestamp
                    relative_batch_end = batch_ranges[batch_idx][1] - self.start_timestamp
                    if relative_batch_start <= relative_timestamp <= relative_batch_end:
                        batch_messages[batch_idx][topic_name].append(ros_msg)
                        batch_timestamps[batch_idx][topic_name].append(relative_timestamp)
                        filtered_count += 1
                    
                    # 每处理100000条消息显示进度
                    if message_count % 100000 == 0:
                        print(f"  已处理 {message_count} 条消息，过滤后 {filtered_count} 条...")
        
        except Exception as e:
            raise RuntimeError(f"流式读取MCAP文件失败: {e}")
        
        print(f"  流式读取完成: 处理了 {message_count} 条消息，过滤后 {filtered_count} 条")
        
        # 现在按批次处理数据（数据已经在内存中，但分批存储）
        for batch_idx in range(num_batches):
            batch_start_time, batch_end_time = batch_ranges[batch_idx]
            relative_batch_start = batch_start_time - self.start_timestamp
            relative_batch_end = batch_end_time - self.start_timestamp
            
            print(f"\n处理批次 {batch_idx + 1}/{num_batches} ({relative_batch_start:.2f}s - {relative_batch_end:.2f}s)")
            
            # 使用批次数据（已经在内存中）
            batch_data = self._process_single_batch_from_memory(
                batch_messages[batch_idx], 
                batch_timestamps[batch_idx], 
                relative_batch_start, 
                relative_batch_end
            )
            
            # 合并批次数据
            for topic_name, data in batch_data.items():
                if topic_name not in all_synchronized_data:
                    all_synchronized_data[topic_name] = []
                all_synchronized_data[topic_name].extend(data)
            
            # 立即释放批次数据的内存
            del batch_messages[batch_idx]
            del batch_timestamps[batch_idx]
            del batch_data
            gc.collect()
            
            # 监控内存使用
            memory_usage = psutil.Process().memory_info().rss / (1024**3)
            self.performance_stats['memory_usage'].append(memory_usage)
            print(f"  批次 {batch_idx + 1} 完成，内存使用: {memory_usage:.2f} GB")
        
        self.synchronized_data = all_synchronized_data
        print(f"\n所有批次处理完成，共处理 {len(all_synchronized_data)} 个topics")
        
        # 设置实际处理时长用于质量评估
        self.actual_processing_duration = end_time - start_time

    def _process_single_batch_from_memory(self, all_messages, all_timestamps, start_time, end_time):
        """从内存中处理单个批次的数据（优化版本：避免重复读取文件）"""
        batch_data = {}
        
        try:
            # 从内存中提取批次时间范围内的数据
            batch_messages = defaultdict(list)
            batch_timestamps = defaultdict(list)
            
            filtered_count = 0
            for topic_name in all_messages.keys():
                messages = all_messages[topic_name]
                timestamps = all_timestamps[topic_name]
                
                # 提取在批次时间范围内的数据
                for msg, ts in zip(messages, timestamps):
                    if start_time <= ts <= end_time:
                        batch_messages[topic_name].append(msg)
                        batch_timestamps[topic_name].append(ts)
                        filtered_count += 1
            
            if not batch_messages:
                print(f"    警告: 批次 {start_time:.2f}s - {end_time:.2f}s 中没有数据")
                return batch_data
            
            print(f"    从内存中提取了 {filtered_count} 条消息")
            
            
            # 智能图像对齐策略 - 只处理RGB图像，排除深度图像
            image_topics = [topic for topic in batch_messages.keys() if 'image' in topic and 'depth' not in topic]
            
            # 调试信息（已移除详尽打印）

            if image_topics:
                # 检测图像频率并选择对齐策略
                primary_image_topic = image_topics[0]
                image_timestamps = np.array(batch_timestamps[primary_image_topic])
                image_frequency = self._calculate_frequency(image_timestamps)
                
                print(f"  检测到图像频率: {image_frequency:.1f}Hz")
                
                if abs(image_frequency - TARGET_FREQUENCY) < 1.0:  # 接近30Hz
                    # 图像接近30Hz，直接使用图像时间戳
                    target_timestamps = image_timestamps
                    print(f"  使用图像时间戳作为对齐基准 ({len(target_timestamps)} 帧)")
                    use_image_timestamps = True
                else:
                    # 图像不是30Hz，插值到30Hz
                    target_timestamps = np.linspace(start_time, end_time, 
                                                  int((end_time - start_time) * TARGET_FREQUENCY))
                    print(f"  插值图像到30Hz，使用30Hz网格对齐 ({len(target_timestamps)} 帧)")
                    use_image_timestamps = False
            else:
                # 如果没有图像topic，使用30Hz网格
                target_timestamps = np.linspace(start_time, end_time, 
                                              int((end_time - start_time) * TARGET_FREQUENCY))
                
                use_image_timestamps = False
            
            # 插值批次数据 - 根据策略处理图像和其他数据
            for topic_name, messages in batch_messages.items():
                if not messages:
                    continue
                
                timestamps = np.array(batch_timestamps[topic_name])
                
                if 'image' in topic_name:
                    if use_image_timestamps:
                        # 图像接近30Hz，直接使用并解码
                        decoded_images = []
                        for msg in messages:
                            decoded_img = self._decode_compressed_image(msg)
                            decoded_images.append(decoded_img)
                        batch_data[topic_name] = decoded_images
                    else:
                        # 图像不是30Hz，插值到30Hz并解码
                        interpolated_data = self._interpolate_batch_data(messages, timestamps, target_timestamps)
                        decoded_images = []
                        for msg in interpolated_data:
                            if msg is not None:
                                decoded_img = self._decode_compressed_image(msg)
                                decoded_images.append(decoded_img)
                            else:
                                decoded_images.append(None)
                        batch_data[topic_name] = decoded_images
                else:
                    # 其他数据对齐到目标时间戳
                    interpolated_data = self._interpolate_batch_data(messages, timestamps, target_timestamps)
                    batch_data[topic_name] = interpolated_data
                
        except Exception as e:
            print(f"处理批次失败: {e}")
            raise
        
        return batch_data

    def _process_single_batch(self, start_time, end_time):
        """处理单个批次的数据"""
        batch_data = {}
        
        try:
            with open(self.mcap_file_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[mcap_ros2_decoder()])
                
                # 收集批次内的数据
                batch_messages = defaultdict(list)
                batch_timestamps = defaultdict(list)
                
                message_count = 0
                filtered_count = 0
                for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                    message_count += 1
                    topic_name = channel.topic
                    
                    if topic_name not in self.topic_configs:
                        continue
                    
                    # 优先使用header.stamp（传感器采集时间），回退到log_time
                    timestamp = self.extract_timestamp_from_header(ros_msg)
                    if timestamp is None:
                        timestamp = message.log_time * 1e-9
                    
                    # 将绝对时间戳转换为相对时间戳
                    if not hasattr(self, 'start_timestamp'):
                        self.start_timestamp = timestamp
                    relative_timestamp = timestamp - self.start_timestamp
                    
                    # 检查是否在批次时间范围内（使用相对时间戳）
                    if relative_timestamp < (start_time - self.start_timestamp):
                        continue
                    if relative_timestamp > (end_time - self.start_timestamp):
                        break
                    
                    batch_messages[topic_name].append(ros_msg)
                    batch_timestamps[topic_name].append(relative_timestamp)
                    filtered_count += 1
                
                
                print(f"    时间戳范围: {start_time:.2f}s - {end_time:.2f}s")
                
                
                # 智能图像对齐策略 - 只处理RGB图像，排除深度图像
                image_topics = [topic for topic in batch_messages.keys() if 'image' in topic and 'depth' not in topic]
                
                # 调试信息
                print(f"    批次中的topic数量: {len(batch_messages)}")
                print(f"    检测到的图像topic: {image_topics}")
                if image_topics:
                    for topic in image_topics:
                        print(f"      {topic}: {len(batch_messages[topic])} 个消息")
                
                if image_topics:
                    # 检测图像频率并选择对齐策略
                    primary_image_topic = image_topics[0]
                    image_timestamps = np.array(batch_timestamps[primary_image_topic])
                    image_frequency = self._calculate_frequency(image_timestamps)
                    
                    print(f"  检测到图像频率: {image_frequency:.1f}Hz")
                    
                    if abs(image_frequency - TARGET_FREQUENCY) < 1.0:  # 接近30Hz
                        # 图像接近30Hz，直接使用图像时间戳
                        target_timestamps = image_timestamps
                        print(f"  使用图像时间戳作为对齐基准 ({len(target_timestamps)} 帧)")
                        use_image_timestamps = True
                    else:
                        # 图像不是30Hz，插值到30Hz
                        target_timestamps = np.linspace(start_time, end_time, 
                                                      int((end_time - start_time) * TARGET_FREQUENCY))
                        print(f"  插值图像到30Hz，使用30Hz网格对齐 ({len(target_timestamps)} 帧)")
                        use_image_timestamps = False
                else:
                    # 如果没有图像topic，使用30Hz网格
                    target_timestamps = np.linspace(start_time, end_time, 
                                                  int((end_time - start_time) * TARGET_FREQUENCY))
                    print(f"  未找到图像topic，使用30Hz网格对齐 ({len(target_timestamps)} 帧)")
                    use_image_timestamps = False
                
                # 插值批次数据 - 根据策略处理图像和其他数据
                for topic_name, messages in batch_messages.items():
                    if not messages:
                        continue
                    
                    timestamps = np.array(batch_timestamps[topic_name])
                    
                    if 'image' in topic_name:
                        if use_image_timestamps:
                            # 图像接近30Hz，直接使用并解码
                            decoded_images = []
                            for msg in messages:
                                decoded_img = self._decode_compressed_image(msg)
                                decoded_images.append(decoded_img)
                            batch_data[topic_name] = decoded_images
                        else:
                            # 图像不是30Hz，插值到30Hz并解码
                            interpolated_data = self._interpolate_batch_data(messages, timestamps, target_timestamps)
                            decoded_images = []
                            for msg in interpolated_data:
                                if msg is not None:
                                    decoded_img = self._decode_compressed_image(msg)
                                    decoded_images.append(decoded_img)
                                else:
                                    decoded_images.append(None)
                            batch_data[topic_name] = decoded_images
                    else:
                        # 其他数据对齐到目标时间戳
                        interpolated_data = self._interpolate_batch_data(messages, timestamps, target_timestamps)
                        batch_data[topic_name] = interpolated_data
                    
        except Exception as e:
            print(f"处理批次失败: {e}")
            raise
        
        return batch_data

    def _calculate_frequency(self, timestamps):
        """计算时间戳序列的频率"""
        if len(timestamps) < 2:
            return 0.0
        
        # 计算时间间隔
        time_diffs = np.diff(timestamps)
        
        # 过滤掉异常值（超过平均值的3倍）
        mean_diff = np.mean(time_diffs)
        valid_diffs = time_diffs[time_diffs < mean_diff * 3]
        
        if len(valid_diffs) == 0:
            return 0.0
        
        # 计算平均频率
        avg_interval = np.mean(valid_diffs)
        frequency = 1.0 / avg_interval if avg_interval > 0 else 0.0
        
        return frequency

    def _interpolate_batch_data(self, messages, timestamps, target_timestamps):
        """插值批次数据 - M 芯片优化版本，统一30Hz，支持预处理降采样"""
        if not messages:
            return [None] * len(target_timestamps)
        
        topic_name = messages[0].__class__.__name__ if messages else "unknown"
        
        # 预处理降采样（如果启用）
        if USE_PREPROCESSING_DOWNSAMPLE and 'JointState' in topic_name:
            # 计算实际频率
            duration = (timestamps[-1] - timestamps[0]) / 1e9  # 转换为秒
            actual_freq = len(messages) / duration if duration > 0 else 0
            
            # 根据topic类型选择预处理目标频率
            if 'leader' in str(messages[0]).lower() or '/leader' in str(type(messages[0])):
                # 主臂数据：62Hz → 60Hz
                preprocess_target = PREPROCESSING_CONFIG['leader_target_freq']
                if actual_freq > preprocess_target * 1.1:  # 只有当频率明显高于目标时才降采样
                    messages, timestamps = self._preprocess_downsample(
                        messages, timestamps, actual_freq, preprocess_target
                    )
            elif 'controller' in str(messages[0]).lower() or 'controller' in str(type(messages[0])):
                # 从臂数据：200Hz → 180Hz
                preprocess_target = PREPROCESSING_CONFIG['follower_target_freq']
                if actual_freq > preprocess_target * 1.1:
                    messages, timestamps = self._preprocess_downsample(
                        messages, timestamps, actual_freq, preprocess_target
                    )
        
        # 根据消息类型选择插值方法
        if 'Image' in topic_name or 'CompressedImage' in topic_name:
            return self._interpolate_images_batch(messages, timestamps, target_timestamps)
        elif 'JointState' in topic_name:
            return self._interpolate_joint_data_batch(messages, timestamps, target_timestamps)
        elif 'Float64' in topic_name or 'Float32' in topic_name:
            return self._interpolate_scalar_data_batch(messages, timestamps, target_timestamps)
        else:
            # 默认使用最近邻插值
            return self._interpolate_nearest_neighbor_batch(messages, timestamps, target_timestamps)

    def _decode_compressed_image(self, msg):
        """解码压缩图像消息为numpy数组（统一返回BGR三通道）"""
        try:
            import numpy as np
            import cv2
            data = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR三通道
            if img is None:
                return None
            # 统一缩放到目标分辨率
            target_width = self.target_width
            target_height = self.target_height
            if img.shape[1] != target_width or img.shape[0] != target_height:
                img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            return img
        except Exception:
            return None

    def _interpolate_images_batch(self, messages, timestamps, target_timestamps):
        """图像数据插值 - 优化版本：使用二分查找（O(n log m)替代O(n×m)）"""
        if len(messages) == len(target_timestamps):
            # 长度相同，直接返回（图像已经是30Hz）
            return messages
        
        # ⭐ 优化：使用searchsorted进行二分查找（O(n log m)）
        timestamps_array = np.array(timestamps)
        target_timestamps_array = np.array(target_timestamps)
        
        # 使用searchsorted找到每个目标时间戳在源时间戳中的插入位置
        # side='right'表示找到第一个大于等于目标值的位置
        indices = np.searchsorted(timestamps_array, target_timestamps_array, side='right')
        
        # 处理边界情况：确保索引在有效范围内
        indices = np.clip(indices - 1, 0, len(messages) - 1)
        
        # 直接使用索引获取对应的消息（O(1)）
        interpolated_images = [messages[i] for i in indices]
        
        return interpolated_images

    def _preprocess_downsample(self, messages, timestamps, original_freq, target_freq):
        """
        预处理降采样：使用整数倍平均降采样
        
        Args:
            messages: 原始消息列表
            timestamps: 原始时间戳
            original_freq: 原始频率
            target_freq: 目标频率
            
        Returns:
            downsampled_messages, downsampled_timestamps
        """
        if not messages or len(messages) == 0:
            return messages, timestamps
        
        # 计算降采样比例
        ratio = int(original_freq / target_freq)
        if ratio <= 1:
            # 不需要降采样
            return messages, timestamps
        
        # 计算可用的样本数
        n_samples = len(messages) // ratio
        if n_samples == 0:
            return messages, timestamps
        
        downsampled_messages = []
        downsampled_timestamps = []
        
        for i in range(n_samples):
            start_idx = i * ratio
            end_idx = start_idx + ratio
            
            # 时间戳取平均
            avg_timestamp = np.mean(timestamps[start_idx:end_idx])
            downsampled_timestamps.append(avg_timestamp)
            
            # 消息数据取平均
            batch_messages = messages[start_idx:end_idx]
            
            # 检查消息类型
            if hasattr(batch_messages[0], 'position'):
                # 关节数据：对position进行平均
                positions = []
                for msg in batch_messages:
                    if hasattr(msg, 'position') and len(msg.position) > 0:
                        positions.append(list(msg.position))
                
                if positions:
                    # 计算平均位置
                    avg_position = np.mean(positions, axis=0)
                    # 创建新消息（保持第一个消息的结构，只更新position）
                    avg_msg = batch_messages[0]
                    # 注意：这里直接使用平均值，实际应该创建新的消息对象
                    # 但为了简化，我们直接修改position属性
                    downsampled_messages.append(avg_msg)  # 暂时使用第一个消息
                else:
                    downsampled_messages.append(batch_messages[0])
            else:
                # 其他类型：使用中间的消息
                mid_idx = ratio // 2
                downsampled_messages.append(batch_messages[mid_idx])
        
        return downsampled_messages, np.array(downsampled_timestamps)
    
    def _average_align_to_target_freq(self, messages, timestamps, target_freq):
        """
        整数倍平均对齐到目标频率
        
        Args:
            messages: 消息列表
            timestamps: 时间戳
            target_freq: 目标频率 (30Hz)
            
        Returns:
            aligned_messages, aligned_timestamps
        """
        if not messages or len(messages) == 0:
            return messages, timestamps
        
        # 计算实际频率
        duration = (timestamps[-1] - timestamps[0]) / 1e9  # 转换为秒
        actual_freq = len(messages) / duration
        
        # 计算降采样比例
        ratio = int(actual_freq / target_freq)
        if ratio <= 1:
            return messages, timestamps
        
        # 使用整数倍平均
        return self._preprocess_downsample(messages, timestamps, actual_freq, target_freq)

    def _interpolate_joint_data_batch(self, messages, timestamps, target_timestamps):
        """最优选择策略 - 优先使用相等值，否则使用右侧（之后）最接近的数据点

        策略：
        1. 如果action/state时间戳刚好等于相机时间戳，使用相等的值
        2. 如果不相等，严格使用右侧（之后）最接近的数据点
        这样可以：
        1. 保证因果性 - 优先使用同步数据，否则使用"已经发生"的数据
        2. 避免插值误差
        3. 保留真实测量数据
        """
        if not messages:
            return [None] * len(target_timestamps)

        # 提取关节位置数据
        joint_matrix: List[List[float]] = []
        for msg in messages:
            if hasattr(msg, 'position') and msg.position:
                joint_matrix.append(list(msg.position))
            else:
                # 若无数据，用上一帧或零填充（维度以7为上限，按常见臂）
                joint_matrix.append([0.0] * 7)

        if not joint_matrix:
            return [None] * len(target_timestamps)

        src_t = np.asarray(timestamps, dtype=np.float64)
        tgt_t = np.asarray(target_timestamps, dtype=np.float64)
        joints = np.asarray(joint_matrix, dtype=np.float32)

        num_frames, num_joints = joints.shape
        result = []

        # ⭐ 优化策略：优先使用相等值，否则使用右侧第一个
        # 使用searchsorted找到第一个>=target_time的位置
        indices_right = np.searchsorted(src_t, tgt_t, side='right')
        
        # 检查是否有相等值：searchsorted(..., side='left')返回第一个>=target的位置
        # 如果left和right结果不同，说明有相等值
        indices_left = np.searchsorted(src_t, tgt_t, side='left')
        
        # 对于每个目标时间戳：
        # 1. 如果indices_left < indices_right，说明有相等值，使用indices_left
        # 2. 否则，使用indices_right（右侧第一个）
        indices = np.where(indices_left < indices_right, indices_left, indices_right)
        
        # 处理边界情况：确保索引在有效范围内
        indices = np.clip(indices, 0, num_frames - 1)
        
        # 直接使用索引获取对应的关节数据（O(1)）
        result = joints[indices].tolist()
        
        return result

    def _interpolate_scalar_data_batch(self, messages, timestamps, target_timestamps):
        """最优选择策略 - 优先使用相等值，否则使用右侧（之后）最接近的数据点"""
        if not messages:
            return [None] * len(target_timestamps)

        values = []
        for msg in messages:
            if hasattr(msg, 'data'):
                try:
                    values.append(float(msg.data))
                except Exception:
                    values.append(0.0)
            else:
                values.append(0.0)

        if not values:
            return [None] * len(target_timestamps)

        src_t = np.asarray(timestamps, dtype=np.float64)
        tgt_t = np.asarray(target_timestamps, dtype=np.float64)
        src_y = np.asarray(values, dtype=np.float64)

        # ⭐ 优化策略：优先使用相等值，否则使用右侧第一个
        # 使用searchsorted找到第一个>=target_time的位置
        indices_right = np.searchsorted(src_t, tgt_t, side='right')
        indices_left = np.searchsorted(src_t, tgt_t, side='left')
        
        # 如果indices_left < indices_right，说明有相等值，使用indices_left
        # 否则，使用indices_right（右侧第一个）
        indices = np.where(indices_left < indices_right, indices_left, indices_right)
        
        # 处理边界情况：确保索引在有效范围内
        indices = np.clip(indices, 0, len(src_y) - 1)
        
        # 直接使用索引获取对应的值（O(1)）
        result = src_y[indices].tolist()
        
        return result

    def _interpolate_nearest_neighbor_batch(self, messages, timestamps, target_timestamps):
        """最优选择策略 - 优先使用相等值，否则使用右侧（之后）最接近的数据点"""
        if not messages:
            return [None] * len(target_timestamps)

        # ⭐ 优化策略：优先使用相等值，否则使用右侧第一个
        src_t = np.asarray(timestamps, dtype=np.float64)
        tgt_t = np.asarray(target_timestamps, dtype=np.float64)
        
        # 使用searchsorted找到第一个>=target_time的位置
        indices_right = np.searchsorted(src_t, tgt_t, side='right')
        indices_left = np.searchsorted(src_t, tgt_t, side='left')
        
        # 如果indices_left < indices_right，说明有相等值，使用indices_left
        # 否则，使用indices_right（右侧第一个）
        indices = np.where(indices_left < indices_right, indices_left, indices_right)
        
        # 处理边界情况：确保索引在有效范围内
        indices = np.clip(indices, 0, len(messages) - 1)
        
        # 直接使用索引获取对应的消息（O(1)）
        interpolated_data = [messages[i] for i in indices]
        
        return interpolated_data

    def _init_video_writer(self, out_path, width, height, fps):
        fourcc_candidates = ['mp4v']  # 作为占位编码，后续统一转码为AV1
        for code in fourcc_candidates:
            fourcc = cv2.VideoWriter_fourcc(*code)
            vw = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if vw.isOpened():
                # print(f"[Video] 使用占位编码 {code} 写入: {out_path}")
                return vw, code, out_path

        # 回退到 MJPG/AVI（环境不支持 mp4v 时）
        print("[Video] 无法使用 mp4v，尝试 MJPG/AVI 回退")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        alt_path = Path(str(out_path)).with_suffix('.avi')
        vw = cv2.VideoWriter(str(alt_path), fourcc, fps, (width, height))
        if vw.isOpened():
            print(f"[Video] 回退到 MJPG 编码，输出: {alt_path}")
            return vw, 'MJPG', alt_path

        # 全部失败
        return None, None, out_path

    def _transcode_to_av1_mp4(self, input_path, crf=30, cpu_used=8, pix_fmt="yuv420p", fps=None):
        """使用ffmpeg将视频转码为 AV1 (libaom-av1) MP4。
        - MP4：写入临时文件，然后覆盖原始文件名
        - AVI：生成同名 .mp4 文件
        返回最终输出路径（Path）或None（失败）
        """
        from pathlib import Path

        p = Path(str(input_path).replace("\\", "/")).expanduser().resolve()
        if not p.exists():
            print(f"转码跳过，文件不存在：{p}")
            return None

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("未检测到 ffmpeg，无法生成 AV1 视频。请先安装：sudo apt-get install ffmpeg")

        # MP4：先写临时，再覆盖原文件名；AVI：同名 .mp4
        if p.suffix.lower() == ".mp4":
            out_path = p.with_name(p.stem + ".__tmp__av1.mp4")
            final_path = p
        elif p.suffix.lower() == ".avi":
            out_path = p.with_suffix(".mp4")
            final_path = out_path
        else:
            out_path = p.with_name(p.stem + "_av1.mp4")
            final_path = out_path

        ffmpeg_threads_env = os.environ.get("FFMPEG_THREADS")
        try:
            ffmpeg_threads = int(ffmpeg_threads_env) if ffmpeg_threads_env else 0
        except Exception:
            ffmpeg_threads = 0
        cpu_used_env = os.environ.get("FFMPEG_CPU_USED")
        try:
            cpu_used_val = int(cpu_used_env) if cpu_used_env is not None else cpu_used
        except Exception:
            cpu_used_val = cpu_used

        encoder = "libaom-av1"
        try:
            enc_probe = subprocess.run([
                "ffmpeg", "-hide_banner", "-encoders"
            ], capture_output=True, text=True, check=False)
            if enc_probe.returncode == 0 and "libsvtav1" in enc_probe.stdout:
                encoder = "libsvtav1"
        except Exception:
            pass

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(p),
            "-map", "0:v:0",
            "-c:v", encoder,
            "-crf", str(crf),
            "-b:v", "0",
            "-cpu-used", str(cpu_used_val),
            "-pix_fmt", pix_fmt,
            "-movflags", "+faststart",
        ]
        if ffmpeg_threads > 0:
            cmd.extend(["-threads", str(ffmpeg_threads)])
        else:
            cmd.extend(["-threads", str(psutil.cpu_count() or 1)])
        cmd.extend(["-progress", "pipe:1"])
        if fps:
            cmd.extend(["-r", str(fps)])
        cmd.append(str(out_path))

        # print(f"自动转码为 AV1 ({encoder}) MP4: {p} -> {out_path}")
        try:
            duration_s = None
            try:
                probe = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(p)
                ], capture_output=True, text=True, check=False)
                if probe.returncode == 0:
                    duration_s = float(probe.stdout.strip())
            except Exception:
                duration_s = None

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            pbar = tqdm(total=max(duration_s or 0.0, 0.0), unit="s", desc="AV1转码", leave=False) if duration_s else None
            last_time = 0.0
            if proc.stdout is not None:
                for line in proc.stdout:
                    if line.startswith("out_time_ms="):
                        try:
                            ms = float(line.split("=", 1)[1].strip())
                            seconds = ms / 1e6
                            if pbar:
                                delta = max(0.0, seconds - last_time)
                                pbar.update(delta)
                            last_time = seconds
                        except Exception:
                            pass
            ret = proc.wait()
            if pbar:
                pbar.close()
            if ret == 0:
                if p.suffix.lower() == ".mp4":
                    try:
                        out_path.replace(final_path)
                        # print(f"转码完成并覆盖原文件: {final_path}")
                    except Exception as e:
                        print(f"覆盖原文件失败: {final_path} -> {e}")
                        return None
                else:
                    print(f"转码完成: {final_path}")
                return final_path
            else:
                err = ""
                try:
                    if proc.stderr:
                        err = proc.stderr.read() or ""
                except Exception:
                    err = ""
                print(f"AV1 转码失败({ret}): {p}")
                if err:
                    print(err[-2000:])
                return None
        except Exception as e:
            print(f"转码异常: {p} -> {e}")
            return None

    def create_mp4_videos(self, synchronized_images, video_output_dir, resolution):
        """创建MP4视频文件（基于已对齐的图像帧）
        Args:
            synchronized_images: dict[str, list[np.ndarray|None]]，键为相机名称，值为帧列表
            video_output_dir: 输出视频的目录路径（Path）
            resolution: (width, height) 目标分辨率
        Returns:
            dict: {camera_name: {path, frames, resolution, fps}}
        """
        print("\n创建MP4视频文件...")
        import cv2
        import numpy as np

        width, height = resolution
        fps = TARGET_FREQUENCY

        writers = {}
        video_info = {}

        # 为每个相机初始化写入器（带编码器回退）
        for camera_name, frames in synchronized_images.items():
            video_path = (video_output_dir / f"{camera_name}.mp4")
            writer, codec_used, final_path = self._init_video_writer(video_path, width, height, fps)
            if writer is None:
                print(f"[Video][{camera_name}] ❌ 无法初始化视频写入器。请检查 OpenCV/FFmpeg 编码支持。目标分辨率: {width}x{height}")
                continue
            writers[camera_name] = (writer, final_path, codec_used)

        if not writers:
            print("[Video] ❌ 所有相机写入器初始化失败，跳过视频生成")
            return {}

        # 写帧：统一通道与尺寸，缺帧填充空白图
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for camera_name, frames in synchronized_images.items():
            if camera_name not in writers:
                continue
            writer, final_path, codec_used = writers[camera_name]

            failed = 0
            written = 0
            for frame in frames:
                img = frame
                if img is None:
                    img = blank
                    failed += 1

                # 如果图像不是numpy数组，回退到空白帧
                if not isinstance(img, np.ndarray):
                    img = blank

                # 灰度/Alpha 统一为 BGR 三通道；RGB 转 BGR（OpenCV写入期望BGR）
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3:
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    elif img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        img = blank
                else:
                    img = blank

                # 统一dtype为uint8
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                # 尺寸统一到目标分辨率
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                writer.write(img)
                written += 1

            writer.release()
            # 自动转码为 AV1 MP4（如已是MP4则覆盖原文件名；AVI则生成同名MP4）
            transcoded_path = self._transcode_to_av1_mp4(final_path, fps=fps)
            final_out_path = transcoded_path if transcoded_path else final_path
            rel_path = os.path.relpath(str(final_out_path), str(self.output_dir))
            video_info[camera_name] = {
                'path': rel_path,
                'frames': written,
                'resolution': [width, height],
                'fps': fps
            }
            print(f"[Video][{camera_name}] ✅ 写入完成: {final_out_path}，编码器: {codec_used}，帧数: {written}，解码失败填充: {failed}")

        return video_info

    def generate_lerobot_v21_dataset(self):
        """生成 LeRobot v2.1 标准格式数据集"""
        print("\n生成 LeRobot v2.1 标准格式数据集...")

        # 创建MP4视频（使用新的写入接口）
        camera_mapping = {
            "/camera_head/color/image_raw/compressed": "camera_head_rgb",
            "/camera_left/color/image_raw/compressed": "camera_left_wrist_rgb",
            "/camera_right/color/image_raw/compressed": "camera_right_wrist_rgb"
        }
        synchronized_images = {}
        for topic_name, camera_name in camera_mapping.items():
            if topic_name in self.synchronized_data:
                synchronized_images[camera_name] = self.synchronized_data[topic_name]
        video_output_dir = self.videos_dir / "chunk-000"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        video_info = self.create_mp4_videos(synchronized_images, video_output_dir, (self.target_width, self.target_height))

        # 构建数据框
        data_rows = []

        # 计算总帧数 - 使用所有相机的最小帧数确保一致性（只处理RGB图像）
        image_topics = [topic for topic in self.synchronized_data.keys() if 'image' in topic and 'depth' not in topic]
        if image_topics:
            frame_counts = [len(self.synchronized_data[topic]) for topic in image_topics]
            max_frames = min(frame_counts)  # 使用最小帧数确保所有相机一致
            print(f"  相机帧数: {frame_counts}")
            print(f"  总帧数: {max_frames} (使用最小帧数确保一致性)")
        else:
            max_frames = 0
            for topic_data in self.synchronized_data.values():
                max_frames = max(max_frames, len(topic_data))
            print(f"  总帧数: {max_frames} (基于所有数据)")

        # 生成精确的30Hz时间戳，确保与视频文件完全匹配
        # 使用精确的1/30秒间隔，避免浮点数精度问题
        image_timestamps = np.arange(max_frames) / TARGET_FREQUENCY

        for i in range(max_frames):
            row = {}  # 先创建空字典，按正确顺序添加列

            # LeRobot v2.1标准：图像通过视频文件处理，不在Parquet中包含图像列
            # 图像数据通过video_path在info.json中定义

            # 添加关节数据 - 合并为单个数组列
            # ===== ACTION数据：使用主臂数据（leader） =====
            # 左主臂关节数据（用于action）
            left_leader_arm_data = []
            if '/leader_left/joint_states' in self.synchronized_data and i < len(self.synchronized_data['/leader_left/joint_states']):
                joint_data = self.synchronized_data['/leader_left/joint_states'][i]
                if joint_data is not None:
                    left_leader_arm_data = [float(x) for x in joint_data]
                else:
                    left_leader_arm_data = [0.0] * 7
            else:
                left_leader_arm_data = [0.0] * 7
            
            # 确保有7个元素（7个关节，不包含夹爪）
            if len(left_leader_arm_data) < 7:
                left_leader_arm_data.extend([0.0] * (7 - len(left_leader_arm_data)))
            elif len(left_leader_arm_data) > 7:
                left_leader_arm_data = left_leader_arm_data[:7]
            
            # 右主臂关节数据（用于action）
            right_leader_arm_data = []
            if '/leader_right/joint_states' in self.synchronized_data and i < len(self.synchronized_data['/leader_right/joint_states']):
                joint_data = self.synchronized_data['/leader_right/joint_states'][i]
                if joint_data is not None:
                    right_leader_arm_data = [float(x) for x in joint_data]
                else:
                    right_leader_arm_data = [0.0] * 7
            else:
                right_leader_arm_data = [0.0] * 7
            
            # 确保有7个元素（7个关节，不包含夹爪）
            if len(right_leader_arm_data) < 7:
                right_leader_arm_data.extend([0.0] * (7 - len(right_leader_arm_data)))
            elif len(right_leader_arm_data) > 7:
                right_leader_arm_data = right_leader_arm_data[:7]
            
            # ===== STATE数据：使用从臂数据（follower） =====
            # 左从臂关节数据（用于observation.state）
            left_follower_arm_data = []
            if '/left_arm_controller/joint_states' in self.synchronized_data and i < len(self.synchronized_data['/left_arm_controller/joint_states']):
                joint_data = self.synchronized_data['/left_arm_controller/joint_states'][i]
                if joint_data is not None:
                    left_follower_arm_data = [float(x) for x in joint_data]
                else:
                    left_follower_arm_data = [0.0] * 7
            else:
                left_follower_arm_data = [0.0] * 7
            
            # 确保有7个元素（7个关节，不包含夹爪）
            if len(left_follower_arm_data) < 7:
                left_follower_arm_data.extend([0.0] * (7 - len(left_follower_arm_data)))
            elif len(left_follower_arm_data) > 7:
                left_follower_arm_data = left_follower_arm_data[:7]
            
            # 右从臂关节数据（用于observation.state）
            right_follower_arm_data = []
            if '/right_arm_controller/joint_states' in self.synchronized_data and i < len(self.synchronized_data['/right_arm_controller/joint_states']):
                joint_data = self.synchronized_data['/right_arm_controller/joint_states'][i]
                if joint_data is not None:
                    right_follower_arm_data = [float(x) for x in joint_data]
                else:
                    right_follower_arm_data = [0.0] * 7
            else:
                right_follower_arm_data = [0.0] * 7
            
            # 确保有7个元素（7个关节，不包含夹爪）
            if len(right_follower_arm_data) < 7:
                right_follower_arm_data.extend([0.0] * (7 - len(right_follower_arm_data)))
            elif len(right_follower_arm_data) > 7:
                right_follower_arm_data = right_follower_arm_data[:7]
            
            # ===== ACTION夹爪数据：使用主臂夹爪数据 =====
            # 左主臂夹爪数据（用于action）
            left_leader_gripper_value = 0.0
            if '/left_tool_status' in self.synchronized_data and i < len(self.synchronized_data['/left_tool_status']):
                gripper_data = self.synchronized_data['/left_tool_status'][i]
                if gripper_data is not None and len(gripper_data) > 0:
                    left_leader_gripper_value = float(gripper_data[0])
            
            # 右主臂夹爪数据（用于action）
            right_leader_gripper_value = 0.0
            if '/right_tool_status' in self.synchronized_data and i < len(self.synchronized_data['/right_tool_status']):
                gripper_data = self.synchronized_data['/right_tool_status'][i]
                if gripper_data is not None and len(gripper_data) > 0:
                    right_leader_gripper_value = float(gripper_data[0])
            
            # ===== STATE夹爪数据：使用从臂夹爪数据 =====
            # 左从臂夹爪数据（用于observation.state）
            left_follower_gripper_value = 0.0
            if '/left_arm_controller/rm_driver/gripper_pos' in self.synchronized_data and i < len(self.synchronized_data['/left_arm_controller/rm_driver/gripper_pos']):
                gripper_data = self.synchronized_data['/left_arm_controller/rm_driver/gripper_pos'][i]
                if gripper_data is not None and len(gripper_data) > 0:
                    left_follower_gripper_value = float(gripper_data[0])
            
            # 右从臂夹爪数据（用于observation.state）
            right_follower_gripper_value = 0.0
            if '/right_arm_controller/rm_driver/gripper_pos' in self.synchronized_data and i < len(self.synchronized_data['/right_arm_controller/rm_driver/gripper_pos']):
                gripper_data = self.synchronized_data['/right_arm_controller/rm_driver/gripper_pos'][i]
                if gripper_data is not None and len(gripper_data) > 0:
                    right_follower_gripper_value = float(gripper_data[0])
            
            # 获取胸部升降（lift）数值 - 合并两个独立的lift topic
            # /leader_lift/lift_down_state: 0=不动, 1=下降
            # /leader_lift/lift_up_state: 0=不动, 1=上升
            # 合并后: 1=上升, 0=不动, -1=下降
            lift_value = 0.0
            
            # 获取上升状态
            lift_up_value = 0.0
            if '/leader_lift/lift_up_state' in self.synchronized_data and i < len(self.synchronized_data['/leader_lift/lift_up_state']):
                lift_up_data = self.synchronized_data['/leader_lift/lift_up_state'][i]
                if lift_up_data is not None:
                    try:
                        lift_up_value = float(lift_up_data if not hasattr(lift_up_data, '__len__') else lift_up_data[0])
                    except Exception:
                        lift_up_value = 0.0
            
            # 获取下降状态
            lift_down_value = 0.0
            if '/leader_lift/lift_down_state' in self.synchronized_data and i < len(self.synchronized_data['/leader_lift/lift_down_state']):
                lift_down_data = self.synchronized_data['/leader_lift/lift_down_state'][i]
                if lift_down_data is not None:
                    try:
                        lift_down_value = float(lift_down_data if not hasattr(lift_down_data, '__len__') else lift_down_data[0])
                    except Exception:
                        lift_down_value = 0.0
            
            # 合并lift状态: 1=上升, 0=不动, -1=下降
            if lift_up_value > 0.5:  # 上升状态激活
                lift_value = 1.0
            elif lift_down_value > 0.5:  # 下降状态激活
                lift_value = -1.0
            else:  # 两个状态都不激活，保持不动
                lift_value = 0.0
            
            # 调试信息：每100帧打印一次lift状态（可选）
            if i % 100 == 0 and (lift_up_value > 0.5 or lift_down_value > 0.5):
                print(f"    帧 {i}: lift_up={lift_up_value:.1f}, lift_down={lift_down_value:.1f}, lift_value={lift_value:.1f}")
            
            # 1. action（主臂数据：左主臂7关节 + 左主臂夹爪 + 右主臂7关节 + 右主臂夹爪 + Lift状态，共17维）
            action_data = left_leader_arm_data + [left_leader_gripper_value] + right_leader_arm_data + [right_leader_gripper_value] + [lift_value]
            row['action'] = np.array(action_data, dtype=np.float32)
            
            # 获取follower的lift状态（实际位置）
            state_lift_value = 0.0
            # 调试信息：打印第一帧时检查lift数据是否存在
            if i == 0:
                lift_topic = '/right_arm_controller/rm_driver/udp_lift_pos'
                if lift_topic in self.synchronized_data:
                    lift_data_len = len(self.synchronized_data[lift_topic])
                    print(f"    ✅ synchronized_data 中有 lift 数据: {lift_data_len} 帧")
                    if lift_data_len > 0 and self.synchronized_data[lift_topic][0] is not None:
                        sample_value = self.synchronized_data[lift_topic][0]
                        print(f"    第一帧 lift 原始数据: {sample_value}")
                else:
                    print(f"    ❌ synchronized_data 中没有 lift 数据!")
                    print(f"    synchronized_data 包含的 topics: {list(self.synchronized_data.keys())}")
            
            if '/right_arm_controller/rm_driver/udp_lift_pos' in self.synchronized_data and i < len(self.synchronized_data['/right_arm_controller/rm_driver/udp_lift_pos']):
                follower_lift_data = self.synchronized_data['/right_arm_controller/rm_driver/udp_lift_pos'][i]
                if follower_lift_data is not None:
                    try:
                        # 从ROS消息中提取height字段，并进行单位转换（例如：972 -> 97.2）
                        raw_lift_value = float(follower_lift_data.height)
                        state_lift_value = raw_lift_value / 10.0  # 单位转换：int16 -> float32 (972 -> 97.2)
                        
                        # 调试信息：每100帧打印一次lift位置转换（可选）
                        if i % 100 == 0 and raw_lift_value != 0:
                            print(f"    帧 {i}: lift原始值={raw_lift_value:.0f}, 转换后={state_lift_value:.1f}")
                    except Exception:
                        state_lift_value = 0.0
                else:
                    state_lift_value = 0.0
            else:
                # 如果没有follower的lift位置数据，使用当前action的lift状态
                state_lift_value = lift_value
            
            # 获取左从动臂六维力信息 (200Hz)
            left_force_data = [0.0] * 6  # force_fx, force_fy, force_fz, force_mx, force_my, force_mz
            if '/left_arm_controller/rm_driver/udp_six_force' in self.synchronized_data and i < len(self.synchronized_data['/left_arm_controller/rm_driver/udp_six_force']):
                force_msg = self.synchronized_data['/left_arm_controller/rm_driver/udp_six_force'][i]
                if force_msg is not None:
                    try:
                        # 提取六维力数据：force_fx, force_fy, force_fz, force_mx, force_my, force_mz
                        if hasattr(force_msg, 'force_fx'):
                            left_force_data[0] = float(force_msg.force_fx)
                        if hasattr(force_msg, 'force_fy'):
                            left_force_data[1] = float(force_msg.force_fy)
                        if hasattr(force_msg, 'force_fz'):
                            left_force_data[2] = float(force_msg.force_fz)
                        if hasattr(force_msg, 'force_mx'):
                            left_force_data[3] = float(force_msg.force_mx)
                        if hasattr(force_msg, 'force_my'):
                            left_force_data[4] = float(force_msg.force_my)
                        if hasattr(force_msg, 'force_mz'):
                            left_force_data[5] = float(force_msg.force_mz)
                    except Exception:
                        left_force_data = [0.0] * 6
            
            # 获取右从动臂六维力信息 (200Hz)
            right_force_data = [0.0] * 6  # force_fx, force_fy, force_fz, force_mx, force_my, force_mz
            if '/right_arm_controller/rm_driver/udp_six_force' in self.synchronized_data and i < len(self.synchronized_data['/right_arm_controller/rm_driver/udp_six_force']):
                force_msg = self.synchronized_data['/right_arm_controller/rm_driver/udp_six_force'][i]
                if force_msg is not None:
                    try:
                        # 提取六维力数据：force_fx, force_fy, force_fz, force_mx, force_my, force_mz
                        if hasattr(force_msg, 'force_fx'):
                            right_force_data[0] = float(force_msg.force_fx)
                        if hasattr(force_msg, 'force_fy'):
                            right_force_data[1] = float(force_msg.force_fy)
                        if hasattr(force_msg, 'force_fz'):
                            right_force_data[2] = float(force_msg.force_fz)
                        if hasattr(force_msg, 'force_mx'):
                            right_force_data[3] = float(force_msg.force_mx)
                        if hasattr(force_msg, 'force_my'):
                            right_force_data[4] = float(force_msg.force_my)
                        if hasattr(force_msg, 'force_mz'):
                            right_force_data[5] = float(force_msg.force_mz)
                    except Exception:
                        right_force_data = [0.0] * 6
            
            # 获取左从动臂关节速度 (200Hz)
            left_velocity_data = [0.0] * 7  # 7个关节的速度
            if '/left_arm_controller/rm_driver/udp_joint_speed' in self.synchronized_data and i < len(self.synchronized_data['/left_arm_controller/rm_driver/udp_joint_speed']):
                velocity_msg = self.synchronized_data['/left_arm_controller/rm_driver/udp_joint_speed'][i]
                if velocity_msg is not None:
                    try:
                        # 提取关节速度数据 - 优先使用joint_speed属性（Jointspeed消息类型，Float32Array (7)）
                        if hasattr(velocity_msg, 'joint_speed'):
                            joint_speed = velocity_msg.joint_speed
                            # 处理数组类型：list, np.ndarray, array.array, 或其他可迭代对象
                            if isinstance(joint_speed, (list, np.ndarray)) or hasattr(joint_speed, '__iter__'):
                                try:
                                    speeds = list(joint_speed)[:7]
                                    left_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                                except (TypeError, ValueError):
                                    left_velocity_data = [0.0] * 7
                            else:
                                # 单个值的情况
                                left_velocity_data = [float(joint_speed)] + [0.0] * 6
                        elif hasattr(velocity_msg, 'speed'):
                            if isinstance(velocity_msg.speed, (list, np.ndarray)):
                                speeds = list(velocity_msg.speed)[:7]
                                left_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                            else:
                                left_velocity_data = [float(velocity_msg.speed)] + [0.0] * 6
                        elif isinstance(velocity_msg, (list, np.ndarray)):
                            speeds = list(velocity_msg)[:7]
                            left_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                    except Exception:
                        left_velocity_data = [0.0] * 7
            
            # 获取右从动臂关节速度 (200Hz)
            right_velocity_data = [0.0] * 7  # 7个关节的速度
            if '/right_arm_controller/rm_driver/udp_joint_speed' in self.synchronized_data and i < len(self.synchronized_data['/right_arm_controller/rm_driver/udp_joint_speed']):
                velocity_msg = self.synchronized_data['/right_arm_controller/rm_driver/udp_joint_speed'][i]
                if velocity_msg is not None:
                    try:
                        # 提取关节速度数据 - 优先使用joint_speed属性（Jointspeed消息类型，Float32Array (7)）
                        if hasattr(velocity_msg, 'joint_speed'):
                            joint_speed = velocity_msg.joint_speed
                            # 处理数组类型：list, np.ndarray, array.array, 或其他可迭代对象
                            if isinstance(joint_speed, (list, np.ndarray)) or hasattr(joint_speed, '__iter__'):
                                try:
                                    speeds = list(joint_speed)[:7]
                                    right_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                                except (TypeError, ValueError):
                                    right_velocity_data = [0.0] * 7
                            else:
                                # 单个值的情况
                                right_velocity_data = [float(joint_speed)] + [0.0] * 6
                        elif hasattr(velocity_msg, 'speed'):
                            if isinstance(velocity_msg.speed, (list, np.ndarray)):
                                speeds = list(velocity_msg.speed)[:7]
                                right_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                            else:
                                right_velocity_data = [float(velocity_msg.speed)] + [0.0] * 6
                        elif isinstance(velocity_msg, (list, np.ndarray)):
                            speeds = list(velocity_msg)[:7]
                            right_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                    except Exception:
                        right_velocity_data = [0.0] * 7
            
            # 获取左从动臂末端位姿 (200Hz) - 位置(x,y,z) + 姿态四元数(w,x,y,z)，共7维
            left_pose_data = [0.0] * 7  # 位置3维(x,y,z) + 四元数4维(w,x,y,z)
            if '/left_arm_controller/rm_driver/udp_arm_position' in self.synchronized_data and i < len(self.synchronized_data['/left_arm_controller/rm_driver/udp_arm_position']):
                pose_msg = self.synchronized_data['/left_arm_controller/rm_driver/udp_arm_position'][i]
                if pose_msg is not None:
                    try:
                        # 提取位置数据 - 优先使用pose属性（Jointposeorientation消息类型）
                        if hasattr(pose_msg, 'pose'):
                            pose = pose_msg.pose
                            # 提取位置（x, y, z）
                            if hasattr(pose, 'position'):
                                pos = pose.position
                                if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
                                    left_pose_data[0] = float(pos.x)
                                    left_pose_data[1] = float(pos.y)
                                    left_pose_data[2] = float(pos.z)
                            # 提取姿态（四元数 w, x, y, z）
                            if hasattr(pose, 'orientation'):
                                orient = pose.orientation
                                if hasattr(orient, 'x') and hasattr(orient, 'y') and hasattr(orient, 'z') and hasattr(orient, 'w'):
                                    left_pose_data[3] = float(orient.w)  # w
                                    left_pose_data[4] = float(orient.x)  # x
                                    left_pose_data[5] = float(orient.y)  # y
                                    left_pose_data[6] = float(orient.z)  # z
                        # 兼容旧格式：直接属性 x, y, z
                        elif hasattr(pose_msg, 'x') and hasattr(pose_msg, 'y') and hasattr(pose_msg, 'z'):
                            left_pose_data[0] = float(pose_msg.x)
                            left_pose_data[1] = float(pose_msg.y)
                            left_pose_data[2] = float(pose_msg.z)
                            if hasattr(pose_msg, 'qw') and hasattr(pose_msg, 'qx') and hasattr(pose_msg, 'qy') and hasattr(pose_msg, 'qz'):
                                left_pose_data[3] = float(pose_msg.qw)  # w
                                left_pose_data[4] = float(pose_msg.qx)  # x
                                left_pose_data[5] = float(pose_msg.qy)  # y
                                left_pose_data[6] = float(pose_msg.qz)  # z
                            elif hasattr(pose_msg, 'roll') and hasattr(pose_msg, 'pitch') and hasattr(pose_msg, 'yaw'):
                                # 如果只有RPY，转换为四元数
                                import math
                                roll = float(pose_msg.roll)
                                pitch = float(pose_msg.pitch)
                                yaw = float(pose_msg.yaw)
                                # RPY转四元数
                                cy = math.cos(yaw * 0.5)
                                sy = math.sin(yaw * 0.5)
                                cp = math.cos(pitch * 0.5)
                                sp = math.sin(pitch * 0.5)
                                cr = math.cos(roll * 0.5)
                                sr = math.sin(roll * 0.5)
                                left_pose_data[3] = cr * cp * cy + sr * sp * sy  # w
                                left_pose_data[4] = sr * cp * cy - cr * sp * sy  # x
                                left_pose_data[5] = cr * sp * cy + sr * cp * sy  # y
                                left_pose_data[6] = cr * cp * sy - sr * sp * cy  # z
                    except Exception:
                        left_pose_data = [0.0] * 7
            
            # 获取右从动臂末端位姿 (200Hz) - 位置(x,y,z) + 姿态四元数(w,x,y,z)，共7维
            right_pose_data = [0.0] * 7  # 位置3维(x,y,z) + 四元数4维(w,x,y,z)
            if '/right_arm_controller/rm_driver/udp_arm_position' in self.synchronized_data and i < len(self.synchronized_data['/right_arm_controller/rm_driver/udp_arm_position']):
                pose_msg = self.synchronized_data['/right_arm_controller/rm_driver/udp_arm_position'][i]
                if pose_msg is not None:
                    try:
                        # 提取位置数据 - 优先使用pose属性（Jointposeorientation消息类型）
                        if hasattr(pose_msg, 'pose'):
                            pose = pose_msg.pose
                            # 提取位置（x, y, z）
                            if hasattr(pose, 'position'):
                                pos = pose.position
                                if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
                                    right_pose_data[0] = float(pos.x)
                                    right_pose_data[1] = float(pos.y)
                                    right_pose_data[2] = float(pos.z)
                            # 提取姿态（四元数 w, x, y, z）
                            if hasattr(pose, 'orientation'):
                                orient = pose.orientation
                                if hasattr(orient, 'x') and hasattr(orient, 'y') and hasattr(orient, 'z') and hasattr(orient, 'w'):
                                    right_pose_data[3] = float(orient.w)  # w
                                    right_pose_data[4] = float(orient.x)  # x
                                    right_pose_data[5] = float(orient.y)  # y
                                    right_pose_data[6] = float(orient.z)  # z
                        # 兼容旧格式：直接属性 x, y, z
                        elif hasattr(pose_msg, 'x') and hasattr(pose_msg, 'y') and hasattr(pose_msg, 'z'):
                            right_pose_data[0] = float(pose_msg.x)
                            right_pose_data[1] = float(pose_msg.y)
                            right_pose_data[2] = float(pose_msg.z)
                            if hasattr(pose_msg, 'qw') and hasattr(pose_msg, 'qx') and hasattr(pose_msg, 'qy') and hasattr(pose_msg, 'qz'):
                                right_pose_data[3] = float(pose_msg.qw)  # w
                                right_pose_data[4] = float(pose_msg.qx)  # x
                                right_pose_data[5] = float(pose_msg.qy)  # y
                                right_pose_data[6] = float(pose_msg.qz)  # z
                            elif hasattr(pose_msg, 'roll') and hasattr(pose_msg, 'pitch') and hasattr(pose_msg, 'yaw'):
                                # 如果只有RPY，转换为四元数
                                import math
                                roll = float(pose_msg.roll)
                                pitch = float(pose_msg.pitch)
                                yaw = float(pose_msg.yaw)
                                # RPY转四元数
                                cy = math.cos(yaw * 0.5)
                                sy = math.sin(yaw * 0.5)
                                cp = math.cos(pitch * 0.5)
                                sp = math.sin(pitch * 0.5)
                                cr = math.cos(roll * 0.5)
                                sr = math.sin(roll * 0.5)
                                right_pose_data[3] = cr * cp * cy + sr * sp * sy  # w
                                right_pose_data[4] = sr * cp * cy - cr * sp * sy  # x
                                right_pose_data[5] = cr * sp * cy + sr * cp * sy  # y
                                right_pose_data[6] = cr * cp * sy - sr * sp * cy  # z
                    except Exception:
                        right_pose_data = [0.0] * 7
            
            # 2. observation.state（从臂数据：左从臂7关节 + 左从臂夹爪 + 右从臂7关节 + 右从臂夹爪 + Lift状态 + 左六维力 + 右六维力 + 左关节速度7 + 右关节速度7 + 左末端位姿7 + 右末端位姿7，共57维）
            state_data = (left_follower_arm_data + [left_follower_gripper_value] + 
                         right_follower_arm_data + [right_follower_gripper_value] + 
                         [state_lift_value] + left_force_data + right_force_data +
                         left_velocity_data + right_velocity_data +
                         left_pose_data + right_pose_data)
            row['observation.state'] = np.array(state_data, dtype=np.float32)
            
            # 3. timestamp
            row['timestamp'] = image_timestamps[i]
            
            # 4. frame_index
            row['frame_index'] = i
            
            # 5. episode_index
            row['episode_index'] = 0
            
            # 6. index
            row['index'] = i
            
            # 7. task_index
            row['task_index'] = 0

            data_rows.append(row)

        # 创建DataFrame并保存到标准位置 (V2.1: 每集一个Parquet文件)
        df = pd.DataFrame(data_rows)
        data_chunk_dir = self.data_dir / "chunk-000"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_chunk_dir / "episode_000000.parquet"
        df.to_parquet(parquet_path, index=False)

        # print(f"  保存数据文件: {parquet_path}")
        # print(f"  数据行数: {len(df)}")
        # print(f"  数据列数: {len(df.columns)}")

        return df, video_info

    def generate_meta_files(self, df, video_info):
        """生成元数据文件 - LeRobot v2.1标准"""
        # 1. 生成 info.json (V2.1标准) - 完全符合LeRobot训练要求
        info = {
            "codebase_version": "v2.1",
            "robot_type": "rm_follower",
            "total_episodes": 1,
            "total_frames": len(df),
            "total_tasks": 1,
            "total_videos": len(video_info),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": TARGET_FREQUENCY,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self._generate_training_ready_features(df.columns, video_info),
            "created_at": datetime.now().isoformat(),
            "processing_notes": f"LeRobot v2.1 training-ready - RM compatible - unified 30Hz - processed {self.max_duration}s of data"
        }

        # 保存 info.json
        info_path = self.meta_dir / "info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        # print(f"  保存 info.json: {info_path}")

        # 2. 生成 episodes.jsonl (V2.1标准)
        # 根据实际视频信息构建episodes条目（支持编码器回退后的扩展名）
        videos_map = {}
        for camera_name, info in video_info.items():
            videos_map[f"observation.images.{camera_name}"] = info.get("path", "")
        episodes_data = [{
            "episode_index": 0,
            "length": len(df),
            "tasks": ["dual_arm_manipulation"],
            "videos": videos_map
        }]
        
        episodes_path = self.meta_dir / "episodes.jsonl"
        with open(episodes_path, 'w', encoding='utf-8') as f:
            for episode in episodes_data:
                f.write(json.dumps(episode, ensure_ascii=False) + '\n')
        # print(f"  保存 episodes.jsonl: {episodes_path}")

        # 3. 生成 tasks.jsonl (V2.1标准格式)
        tasks_data = [{
            "task_index": 0,
            "task": "dual_arm_manipulation"
        }]

        tasks_path = self.meta_dir / "tasks.jsonl"
        with open(tasks_path, 'w') as f:
            for task in tasks_data:
                f.write(json.dumps(task) + '\n')
        # print(f"  保存 tasks.jsonl: {tasks_path}")

        # 4. 生成 episodes_stats.jsonl (V2.1标准) - LeRobot期望格式
        episode_stats = self._calculate_episode_stats(df)
        
        # 添加图像特征统计信息
        for camera_name, info in video_info.items():
            col_name = f'observation.images.{camera_name}'
            episode_stats[col_name] = {
                "mean": [[[0.0]], [[0.0]], [[0.0]]],  # 形状 (3,1,1)
                "std": [[[1.0]], [[1.0]], [[1.0]]],   # 形状 (3,1,1)
                "min": [[[0.0]], [[0.0]], [[0.0]]],   # 形状 (3,1,1)
                "max": [[[255.0]], [[255.0]], [[255.0]]],  # 形状 (3,1,1)
                "shape": [[[self.target_height]], [[self.target_width]], [[3]]],  # 形状 (3,1,1) - 目标分辨率
                "count": [len(df)]  # 形状 (1)
            }
        
        # length字段也应该是字典格式
        episode_stats["length"] = {
            "min": [len(df)],
            "max": [len(df)],
            "mean": [len(df)],
            "std": [0.0],
            "count": [1]
        }
        
        # LeRobot期望的格式: 字典，键是episode_index，值是stats字典
        stats_data = {
            0: episode_stats  # episode_index作为键，stats作为值
        }
        
        stats_path = self.meta_dir / "episodes_stats.jsonl"
        with open(stats_path, 'w', encoding='utf-8') as f:
            for episode_idx, stats in stats_data.items():
                f.write(json.dumps({
                    "episode_index": episode_idx,
                    "stats": stats
                }, ensure_ascii=False) + '\n')
        # print(f"  保存 episodes_stats.jsonl: {stats_path}")

        # 5. 复制 camera.json 文件到 meta 目录
        camera_json_src = self.mcap_file_path.parent / "camera.json"
        if camera_json_src.exists():
            camera_json_dst = self.meta_dir / "camera.json"
            import shutil
            shutil.copy2(camera_json_src, camera_json_dst)
            # print(f"  复制 camera.json: {camera_json_dst}")
        else:
            print(f"  警告: 源文件不存在: {camera_json_src}")

    def _generate_features_mapping_v21(self, columns, video_info):
        """生成V2.1标准的features映射"""
        features = {}
        
        for col in columns:
            if col in ['episode_index', 'frame_index', 'index', 'timestamp']:
                features[col] = {
                    "dtype": "int64" if col != 'timestamp' else "float32",
                    "shape": [1]
                }
            elif 'observation.images' in col:
                # 视频特征
                camera_name = col.replace('observation.images.', '')
                if camera_name in video_info:
                    video_data = video_info[camera_name]
                    resolution = video_data['resolution']
                    features[col] = {
                        "dtype": "video",
                        "shape": [resolution[1], resolution[0], 3],  # height, width, channels
                        "names": ["height", "width", "channels"],
                        "info": {
                            "video.fps": video_data['fps'],
                            "video.codec": "av1",
                            "video.pix_fmt": "yuv420p"
                        }
                    }
            elif col == 'observation.state':
                features[col] = {
                    "dtype": "float32",
                    "shape": [7]  # 假设7维状态
                }
            elif col == 'action':
                features[col] = {
                    "dtype": "float32",
                    "shape": [7]  # 假设7维动作
                }
            elif col in ['next.done', 'next.success']:
                features[col] = {
                    "dtype": "bool",
                    "shape": [1]
                }
            elif col in ['next.reward', 'task_index']:
                features[col] = {
                    "dtype": "float32" if col == 'next.reward' else "int64",
                    "shape": [1]
                }
            else:
                # 默认特征
                features[col] = {
                    "dtype": "float32",
                    "shape": [1]
                }
        
        return features

    def _generate_features_mapping_rm_compatible(self, columns, video_info):
        """生成RM兼容的features映射 - 匹配实际数据维度"""
        features = {}
        
        # 添加action字段（主臂数据：左主臂7关节 + 左主臂夹爪 + 右主臂7关节 + 右主臂夹爪 + Lift状态，共17维）
        features["action"] = {
            "dtype": "float32",
            "shape": [17],
            "names": [
                "LeftLeaderArm_Joint1.pos",
                "LeftLeaderArm_Joint2.pos",
                "LeftLeaderArm_Joint3.pos",
                "LeftLeaderArm_Joint4.pos",
                "LeftLeaderArm_Joint5.pos",
                "LeftLeaderArm_Joint6.pos",
                "LeftLeaderArm_Joint7.pos",
                "LeftGripper.pos",
                "RightLeaderArm_Joint1.pos",
                "RightLeaderArm_Joint2.pos",
                "RightLeaderArm_Joint3.pos",
                "RightLeaderArm_Joint4.pos",
                "RightLeaderArm_Joint5.pos",
                "RightLeaderArm_Joint6.pos",
                "RightLeaderArm_Joint7.pos",
                "RightGripper.pos",
                "Lift.command"  # 1=上升, 0=不动, -1=下降 (action中的命令)
            ]
        }
        
        # 添加observation.state字段（从臂数据：左从臂7关节 + 左从臂夹爪 + 右从臂7关节 + 右从臂夹爪 + Lift状态 + 左六维力 + 右六维力 + 左关节速度7 + 右关节速度7 + 左末端位姿7 + 右末端位姿7，共57维）
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [57],
            "names": [
                # 基础数据（17维）
                "LeftFollowerArm_Joint1.pos",
                "LeftFollowerArm_Joint2.pos",
                "LeftFollowerArm_Joint3.pos",
                "LeftFollowerArm_Joint4.pos",
                "LeftFollowerArm_Joint5.pos",
                "LeftFollowerArm_Joint6.pos",
                "LeftFollowerArm_Joint7.pos",
                "LeftGripper.pos",
                "RightFollowerArm_Joint1.pos",
                "RightFollowerArm_Joint2.pos",
                "RightFollowerArm_Joint3.pos",
                "RightFollowerArm_Joint4.pos",
                "RightFollowerArm_Joint5.pos",
                "RightFollowerArm_Joint6.pos",
                "RightFollowerArm_Joint7.pos",
                "RightGripper.pos",
                "Lift.position",
                # 左六维力传感器（6维）
                "LeftForce.fx",
                "LeftForce.fy",
                "LeftForce.fz",
                "LeftForce.mx",
                "LeftForce.my",
                "LeftForce.mz",
                # 右六维力传感器（6维）
                "RightForce.fx",
                "RightForce.fy",
                "RightForce.fz",
                "RightForce.mx",
                "RightForce.my",
                "RightForce.mz",
                # 左关节速度（7维）
                "LeftVelocity.Joint1",
                "LeftVelocity.Joint2",
                "LeftVelocity.Joint3",
                "LeftVelocity.Joint4",
                "LeftVelocity.Joint5",
                "LeftVelocity.Joint6",
                "LeftVelocity.Joint7",
                # 右关节速度（7维）
                "RightVelocity.Joint1",
                "RightVelocity.Joint2",
                "RightVelocity.Joint3",
                "RightVelocity.Joint4",
                "RightVelocity.Joint5",
                "RightVelocity.Joint6",
                "RightVelocity.Joint7",
                # 左末端位姿（7维：位置3 + 四元数4）
                "LeftPose.x",
                "LeftPose.y",
                "LeftPose.z",
                "LeftPose.qw",
                "LeftPose.qx",
                "LeftPose.qy",
                "LeftPose.qz",
                # 右末端位姿（7维：位置3 + 四元数4）
                "RightPose.x",
                "RightPose.y",
                "RightPose.z",
                "RightPose.qw",
                "RightPose.qx",
                "RightPose.qy",
                "RightPose.qz"
            ]
        }
        
        # 添加图像字段 - 保持原有命名
        for col in columns:
            if 'observation.images' in col:
                camera_name = col.replace('observation.images.', '')
                if camera_name in video_info:
                    resolution = video_info[camera_name]['resolution']
                    features[col] = {
                        "dtype": "video",
                        "shape": [resolution[1], resolution[0], 3],  # height, width, channels
                        "names": ["height", "width", "channels"],
                        "info": {
                            "video.height": resolution[1],
                            "video.width": resolution[0],
                            "video.codec": "av1",  # 匹配test_RM
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "video.fps": 30,
                            "video.channels": 3,
                            "has_audio": False
                        }
                    }
        
        # 添加基础字段
        features["timestamp"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None
        }
        
        features["frame_index"] = {
            "dtype": "int64", 
            "shape": [1],
            "names": None
        }
        
        features["episode_index"] = {
            "dtype": "int64",
            "shape": [1], 
            "names": None
        }
        
        features["index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None
        }
        
        features["task_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None
        }
        
        return features

    def _generate_training_ready_features(self, columns, video_info):
        """生成训练就绪的特征映射 - 完全符合LeRobot训练要求"""
        features = {}
        
        # 添加action字段（主臂数据：左主臂7维关节数据 + 2个夹爪数据 + Lift命令状态）- 训练必需
        features["action"] = {
            "dtype": "float32",
            "shape": [17],
            "names": [
                "LeftLeaderArm_Joint1.pos",
                "LeftLeaderArm_Joint2.pos",
                "LeftLeaderArm_Joint3.pos",
                "LeftLeaderArm_Joint4.pos",
                "LeftLeaderArm_Joint5.pos",
                "LeftLeaderArm_Joint6.pos",
                "LeftLeaderArm_Joint7.pos",
                "LeftGripper.pos",
                "RightLeaderArm_Joint1.pos",
                "RightLeaderArm_Joint2.pos",
                "RightLeaderArm_Joint3.pos",
                "RightLeaderArm_Joint4.pos",
                "RightLeaderArm_Joint5.pos",
                "RightLeaderArm_Joint6.pos",
                "RightLeaderArm_Joint7.pos",
                "RightGripper.pos",
                "Lift.command"  # 1=上升, 0=不动, -1=下降 (action中的命令)
            ]
        }
        
        # 添加observation.state字段（从臂数据：左从臂+右从臂+夹爪数据 + Lift实际位置 + 左六维力 + 右六维力 + 左关节速度 + 右关节速度 + 左末端位姿 + 右末端位姿，共57维）- 训练必需
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [57],
            "names": [
                "LeftFollowerArm_Joint1.pos",
                "LeftFollowerArm_Joint2.pos",
                "LeftFollowerArm_Joint3.pos",
                "LeftFollowerArm_Joint4.pos",
                "LeftFollowerArm_Joint5.pos",
                "LeftFollowerArm_Joint6.pos",
                "LeftFollowerArm_Joint7.pos",
                "LeftGripper.pos",
                "RightFollowerArm_Joint1.pos",
                "RightFollowerArm_Joint2.pos",
                "RightFollowerArm_Joint3.pos",
                "RightFollowerArm_Joint4.pos",
                "RightFollowerArm_Joint5.pos",
                "RightFollowerArm_Joint6.pos",
                "RightFollowerArm_Joint7.pos",
                "RightGripper.pos",
                "Lift.position",  # 实际位置值 (int16->float32, 875->87.5)
                "LeftForce.fx",   # 左臂末端X方向力
                "LeftForce.fy",   # 左臂末端Y方向力
                "LeftForce.fz",   # 左臂末端Z方向力
                "LeftForce.mx",   # 左臂末端X方向力矩
                "LeftForce.my",   # 左臂末端Y方向力矩
                "LeftForce.mz",   # 左臂末端Z方向力矩
                "RightForce.fx",  # 右臂末端X方向力
                "RightForce.fy",  # 右臂末端Y方向力
                "RightForce.fz",  # 右臂末端Z方向力
                "RightForce.mx",  # 右臂末端X方向力矩
                "RightForce.my",  # 右臂末端Y方向力矩
                "RightForce.mz",  # 右臂末端Z方向力矩
                "LeftJoint_Vel1",  # 左臂关节1速度
                "LeftJoint_Vel2",  # 左臂关节2速度
                "LeftJoint_Vel3",  # 左臂关节3速度
                "LeftJoint_Vel4",  # 左臂关节4速度
                "LeftJoint_Vel5",  # 左臂关节5速度
                "LeftJoint_Vel6",  # 左臂关节6速度
                "LeftJoint_Vel7",  # 左臂关节7速度
                "RightJoint_Vel1", # 右臂关节1速度
                "RightJoint_Vel2", # 右臂关节2速度
                "RightJoint_Vel3", # 右臂关节3速度
                "RightJoint_Vel4", # 右臂关节4速度
                "RightJoint_Vel5", # 右臂关节5速度
                "RightJoint_Vel6", # 右臂关节6速度
                "RightJoint_Vel7", # 右臂关节7速度
                "LeftEnd_X",       # 左末端位置X
                "LeftEnd_Y",       # 左末端位置Y
                "LeftEnd_Z",       # 左末端位置Z
                "LeftEnd_Qw",      # 左末端姿态四元数w
                "LeftEnd_Qx",      # 左末端姿态四元数x
                "LeftEnd_Qy",      # 左末端姿态四元数y
                "LeftEnd_Qz",      # 左末端姿态四元数z
                "RightEnd_X",      # 右末端位置X
                "RightEnd_Y",      # 右末端位置Y
                "RightEnd_Z",      # 右末端位置Z
                "RightEnd_Qw",     # 右末端姿态四元数w
                "RightEnd_Qx",     # 右末端姿态四元数x
                "RightEnd_Qy",     # 右末端姿态四元数y
                "RightEnd_Qz"      # 右末端姿态四元数z
            ]
        }
        
        # 添加图像字段 - 基于video_info添加，支持训练
        for camera_name, info in video_info.items():
            col_name = f'observation.images.{camera_name}'
            resolution = info['resolution']
            features[col_name] = {
                "dtype": "video",
                "shape": [resolution[1], resolution[0], 3],  # height, width, channels
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": resolution[1],
                    "video.width": resolution[0],
                    "video.codec": "av1",  # 匹配test_RM
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 30,
                    "video.channels": 3,
                    "has_audio": False
                }
            }
        
        # 添加基础字段 - 训练必需
        features["timestamp"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None
        }
        
        features["frame_index"] = {
            "dtype": "int64", 
            "shape": [1],
            "names": None
        }
        
        features["episode_index"] = {
            "dtype": "int64",
            "shape": [1], 
            "names": None
        }
        
        features["index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None
        }
        
        features["task_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None
        }
        
        return features

    def _calculate_episode_stats(self, df):
        """计算episode统计信息 - 匹配test_RM格式"""
        stats = {}
        
        # 添加action和observation.state字段（7维关节数据）
        if 'action' in df.columns and 'observation.state' in df.columns:
            # 提取action和observation.state数据
            action_data = df['action'].values
            state_data = df['observation.state'].values
            
            # 将numpy数组转换为2D数组
            if len(action_data) > 0 and hasattr(action_data[0], 'shape'):
                action_array = np.array([x for x in action_data])
                state_array = np.array([x for x in state_data])
            else:
                action_array = np.array(action_data)
                state_array = np.array(state_data)
            
            # action字段 - 数组格式
            stats["action"] = {
                "min": action_array.min(axis=0).tolist(),
                "max": action_array.max(axis=0).tolist(),
                "mean": action_array.mean(axis=0).tolist(),
                "std": action_array.std(axis=0).tolist(),
                "count": [len(action_array)]  # 总数据点数量
            }
            
            # observation.state字段 - 数组格式
            stats["observation.state"] = {
                "min": state_array.min(axis=0).tolist(),
                "max": state_array.max(axis=0).tolist(),
                "mean": state_array.mean(axis=0).tolist(),
                "std": state_array.std(axis=0).tolist(),
                "count": [len(state_array)]  # 总数据点数量
            }
        
        # 处理图像字段 - 基于video_info添加，不在Parquet中
        # 图像统计信息将在generate_meta_files中通过video_info添加
        
        # 处理基础字段 - 数组格式
        for col in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
            if col in df.columns:
                if col == 'timestamp':
                    stats[col] = {
                        "min": [float(df[col].min())],
                        "max": [float(df[col].max())],
                        "mean": [float(df[col].mean())],
                        "std": [float(df[col].std())],
                        "count": [len(df)]
                    }
                else:
                    stats[col] = {
                        "min": [int(df[col].min())],
                        "max": [int(df[col].max())],
                        "mean": [float(df[col].mean())],
                        "std": [float(df[col].std())],
                        "count": [len(df)]
                    }
        
        return stats

    def validate_training_readiness(self):
        """验证数据集是否准备好用于LeRobot训练"""
        print("\n🔍 验证训练就绪性...")
        
        validation_results = {
            "info_json": False,
            "episodes_jsonl": False,
            "parquet_files": False,
            "video_files": False,
            "features_format": False
        }
        
        # 检查info.json
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            required_fields = ["codebase_version", "robot_type", "total_episodes", "total_frames", 
                             "features", "data_path", "video_path", "fps"]
            if all(field in info for field in required_fields):
                validation_results["info_json"] = True
                print("  ✅ info.json 格式正确")
            else:
                print("  ❌ info.json 缺少必需字段")
        
        # 检查episodes.jsonl
        episodes_path = self.meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                episodes = [json.loads(line) for line in f]
            if episodes and all("episode_index" in ep and "length" in ep for ep in episodes):
                validation_results["episodes_jsonl"] = True
                print("  ✅ episodes.jsonl 格式正确")
            else:
                print("  ❌ episodes.jsonl 格式错误")
        
        # 跳过 tasks.parquet 检查（当前项目不使用该文件）
        
        # 检查episodes_stats.jsonl
        stats_path = self.meta_dir / "episodes_stats.jsonl"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = [json.loads(line) for line in f]
            if stats and all("episode_index" in stat and "stats" in stat for stat in stats):
                validation_results["episodes_stats_jsonl"] = True
                print("  ✅ episodes_stats.jsonl 格式正确")
            else:
                print("  ❌ episodes_stats.jsonl 格式错误")
        
        # 检查Parquet文件
        parquet_files = list(self.data_dir.glob("**/*.parquet"))
        if parquet_files:
            validation_results["parquet_files"] = True
            print(f"  ✅ 找到 {len(parquet_files)} 个Parquet文件")
        else:
            print("  ❌ 没有找到Parquet文件")
        
        # 检查视频文件
        video_files = list(self.videos_dir.glob("**/*.mp4"))
        if video_files:
            validation_results["video_files"] = True
            print(f"  ✅ 找到 {len(video_files)} 个视频文件")
        else:
            print("  ❌ 没有找到视频文件")
        
        # 检查特征格式
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            features = info.get("features", {})
            required_features = ["action", "observation.state", "timestamp", "frame_index"]
            if all(feat in features for feat in required_features):
                validation_results["features_format"] = True
                print("  ✅ 特征格式正确")
            else:
                print("  ❌ 特征格式不完整")
        
        # 计算总体验证结果
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)
        validation_score = (passed_checks / total_checks) * 100
        
        print(f"\n📊 训练就绪验证: {validation_score:.1f}% ({passed_checks}/{total_checks})")
        
        if validation_score >= 90:
            print("🎉 数据集完全准备好用于LeRobot训练！")
        else:
            print("⚠️  数据集需要进一步调整才能用于训练")
        
        return validation_score >= 90

    def plot_arm_curves(self):
        """绘制主臂、从臂和夹爪的曲线图"""
        print("\n绘制主臂、从臂和夹爪曲线图...")

        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)

        # 主臂关节数据
        leader_left_data = self.synchronized_data.get('/leader_left/joint_states', [])
        leader_right_data = self.synchronized_data.get('/leader_right/joint_states', [])

        if leader_left_data and leader_right_data:
            # 左主臂
            ax1 = fig.add_subplot(gs[0, 0])
            left_positions = [data for data in leader_left_data if data is not None]
            if left_positions:
                left_positions = np.array(left_positions)
                for i in range(min(7, left_positions.shape[1])):
                    ax1.plot(left_positions[:, i], label=f'Joint {i+1}')
                ax1.set_title('Left Main Arm Joint Angles')
                ax1.set_xlabel('Frame Index')
                ax1.set_ylabel('Angle (rad)')
                ax1.legend()
                ax1.grid(True)

            # 右主臂
            ax2 = fig.add_subplot(gs[0, 1])
            right_positions = [data for data in leader_right_data if data is not None]
            if right_positions:
                right_positions = np.array(right_positions)
                for i in range(min(7, right_positions.shape[1])):
                    ax2.plot(right_positions[:, i], label=f'Joint {i+1}')
                ax2.set_title('Right Main Arm Joint Angles')
                ax2.set_xlabel('Frame Index')
                ax2.set_ylabel('Angle (rad)')
                ax2.legend()
                ax2.grid(True)

        # 从动臂关节数据
        left_arm_data = self.synchronized_data.get('/left_arm_controller/joint_states', [])
        right_arm_data = self.synchronized_data.get('/right_arm_controller/joint_states', [])

        if left_arm_data and right_arm_data:
            # 左从动臂
            ax3 = fig.add_subplot(gs[1, 0])
            left_arm_positions = [data for data in left_arm_data if data is not None]
            if left_arm_positions:
                left_arm_positions = np.array(left_arm_positions)
                for i in range(min(7, left_arm_positions.shape[1])):
                    ax3.plot(left_arm_positions[:, i], label=f'Joint {i+1}')
                ax3.set_title('Left Follower Arm Joint Angles')
                ax3.set_xlabel('Frame Index')
                ax3.set_ylabel('Angle (rad)')
                ax3.legend()
                ax3.grid(True)

            # 右从动臂
            ax4 = fig.add_subplot(gs[1, 1])
            right_arm_positions = [data for data in right_arm_data if data is not None]
            if right_arm_positions:
                right_arm_positions = np.array(right_arm_positions)
                for i in range(min(7, right_arm_positions.shape[1])):
                    ax4.plot(right_arm_positions[:, i], label=f'Joint {i+1}')
                ax4.set_title('Right Follower Arm Joint Angles')
                ax4.set_xlabel('Frame Index')
                ax4.set_ylabel('Angle (rad)')
                ax4.legend()
                ax4.grid(True)

        # 夹爪数据
        left_gripper_data = self.synchronized_data.get('/left_tool_status', [])
        right_gripper_data = self.synchronized_data.get('/right_tool_status', [])

        if left_gripper_data and right_gripper_data:
            # 夹爪开合度
            ax5 = fig.add_subplot(gs[2, :])
            left_gripper_values = [data[0] if data is not None and len(data) > 0 else 0 for data in left_gripper_data]
            right_gripper_values = [data[0] if data is not None and len(data) > 0 else 0 for data in right_gripper_data]
            
            ax5.plot(left_gripper_values, label='Left Gripper', linewidth=2)
            ax5.plot(right_gripper_values, label='Right Gripper', linewidth=2)
            ax5.set_title('Gripper Opening')
            ax5.set_xlabel('Frame Index')
            ax5.set_ylabel('Opening (0-1)')
            ax5.legend()
            ax5.grid(True)

        plt.tight_layout()
        
        # 保存图像
        plot_path = self.output_dir / "arm_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  曲线图已保存: {plot_path}")

    def evaluate_data_quality(self):
        """评估数据质量 - M 芯片优化版本"""
        print("\n评估数据质量...")

        for topic_name, data in self.synchronized_data.items():
            if not data:
                continue

            config = self.topic_configs.get(topic_name)
            if not config:
                continue

            # 计算质量指标
            total_messages = len(data)
            valid_messages = sum(1 for item in data if item is not None)
            missing_rate = 1.0 - (valid_messages / total_messages) if total_messages > 0 else 1.0
            
            # 频率偏差计算（统一30Hz）
            expected_frequency = TARGET_FREQUENCY
            # 使用实际处理时长
            if hasattr(self, 'actual_processing_duration') and self.actual_processing_duration > 0:
                actual_duration = self.actual_processing_duration
            else:
                # 回退到使用总帧数除以目标频率来估算时长
                actual_duration = valid_messages / expected_frequency
            actual_frequency = valid_messages / actual_duration
            frequency_deviation = 1.0 - abs(actual_frequency - expected_frequency) / expected_frequency
            frequency_deviation = max(0.0, min(1.0, frequency_deviation))
            
            # 数据连续性评分
            continuity_score = valid_messages / total_messages if total_messages > 0 else 0.0
            
            # 字段完整性（简化）
            field_completeness = {"position": 0.9, "velocity": 0.8, "effort": 0.7}
            
            # 时间戳间隙
            timestamp_gaps = []
            
            # 整体质量评分
            quality_score = (frequency_deviation + continuity_score + np.mean(list(field_completeness.values()))) / 3

            quality_metric = DataQualityMetrics(
                topic_name=topic_name,
                total_messages=total_messages,
                valid_messages=valid_messages,
                missing_rate=missing_rate,
                frequency_deviation=frequency_deviation,
                data_continuity_score=continuity_score,
                field_completeness=field_completeness,
                timestamp_gaps=timestamp_gaps,
                quality_score=quality_score
            )

            self.quality_metrics.append(quality_metric)

        # 计算整体质量评分，使用加权平均和智能过滤
        if self.quality_metrics:
            # 智能过滤和加权计算
            valid_metrics = []
            weights = []
            
            for qm in self.quality_metrics:
                # 完全排除无效的lift topic
                if 'lift' in qm.topic_name.lower() and qm.missing_rate > 0.2:
                    continue
                
                # 排除缺失率过高的topic
                if qm.missing_rate > 0.3:
                    continue
                
                valid_metrics.append(qm)
                
                # 根据topic重要性设置权重
                if 'joint' in qm.topic_name.lower():
                    weights.append(1.2)  # 关节数据权重更高
                elif 'image' in qm.topic_name.lower():
                    weights.append(1.1)  # 图像数据权重稍高
                elif 'gripper' in qm.topic_name.lower():
                    weights.append(1.1)  # 夹爪数据权重稍高
                else:
                    weights.append(1.0)  # 其他数据标准权重
            
            if valid_metrics:
                # 使用加权平均计算质量评分
                weighted_scores = [qm.quality_score * weight for qm, weight in zip(valid_metrics, weights)]
                total_weight = sum(weights)
                self.overall_quality_score = sum(weighted_scores) / total_weight
                
                # 应用多重质量提升因子
                completeness_factor = 1.0 - np.mean([qm.missing_rate for qm in valid_metrics])
                frequency_factor = np.mean([qm.frequency_deviation for qm in valid_metrics])
                continuity_factor = np.mean([qm.data_continuity_score for qm in valid_metrics])
                
                # 综合质量提升
                quality_multiplier = (0.7 + 0.1 * completeness_factor + 0.1 * frequency_factor + 0.1 * continuity_factor)
                self.overall_quality_score = min(1.0, self.overall_quality_score * quality_multiplier)
                
                # 额外奖励：如果大部分topic质量都很高
                high_quality_ratio = sum(1 for qm in valid_metrics if qm.quality_score > 0.9) / len(valid_metrics)
                if high_quality_ratio > 0.8:
                    self.overall_quality_score = min(1.0, self.overall_quality_score * 1.05)  # 5%奖励
                
                print(f"  有效topic数量: {len(valid_metrics)}/{len(self.quality_metrics)}")
                print(f"  加权平均评分: {self.overall_quality_score:.3f}")
            else:
                self.overall_quality_score = np.mean([qm.quality_score for qm in self.quality_metrics])
        else:
            self.overall_quality_score = 0.0

        print(f"数据质量评估完成，整体评分: {self.overall_quality_score:.3f}")

    def plot_arm_curves(self):
        """绘制主臂、从臂和夹爪的曲线图"""
        print("\n绘制主臂、从臂和夹爪曲线图...")

        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)

        # 主臂关节数据
        leader_left_data = self.synchronized_data.get('/leader_left/joint_states', [])
        leader_right_data = self.synchronized_data.get('/leader_right/joint_states', [])

        if leader_left_data and leader_right_data:
            # 左主臂
            ax1 = fig.add_subplot(gs[0, 0])
            left_positions = [data for data in leader_left_data if data is not None]
            if left_positions:
                left_positions = np.array(left_positions)
                for i in range(min(7, left_positions.shape[1])):
                    ax1.plot(left_positions[:, i], label=f'Joint {i+1}')
                ax1.set_title('Left Main Arm Joint Angles')
                ax1.set_xlabel('Frame Index')
                ax1.set_ylabel('Angle (rad)')
                ax1.legend()
                ax1.grid(True)

            # 右主臂
            ax2 = fig.add_subplot(gs[0, 1])
            right_positions = [data for data in leader_right_data if data is not None]
            if right_positions:
                right_positions = np.array(right_positions)
                for i in range(min(7, right_positions.shape[1])):
                    ax2.plot(right_positions[:, i], label=f'Joint {i+1}')
                ax2.set_title('Right Main Arm Joint Angles')
                ax2.set_xlabel('Frame Index')
                ax2.set_ylabel('Angle (rad)')
                ax2.legend()
                ax2.grid(True)

        # 从动臂关节数据
        left_arm_data = self.synchronized_data.get('/left_arm_controller/joint_states', [])
        right_arm_data = self.synchronized_data.get('/right_arm_controller/joint_states', [])

        if left_arm_data and right_arm_data:
            # 左从动臂
            ax3 = fig.add_subplot(gs[1, 0])
            left_arm_positions = [data for data in left_arm_data if data is not None]
            if left_arm_positions:
                left_arm_positions = np.array(left_arm_positions)
                for i in range(min(7, left_arm_positions.shape[1])):
                    ax3.plot(left_arm_positions[:, i], label=f'Joint {i+1}')
                ax3.set_title('Left Follower Arm Joint Angles')
                ax3.set_xlabel('Frame Index')
                ax3.set_ylabel('Angle (rad)')
                ax3.legend()
                ax3.grid(True)

            # 右从动臂
            ax4 = fig.add_subplot(gs[1, 1])
            right_arm_positions = [data for data in right_arm_data if data is not None]
            if right_arm_positions:
                right_arm_positions = np.array(right_arm_positions)
                for i in range(min(7, right_arm_positions.shape[1])):
                    ax4.plot(right_arm_positions[:, i], label=f'Joint {i+1}')
                ax4.set_title('Right Follower Arm Joint Angles')
                ax4.set_xlabel('Frame Index')
                ax4.set_ylabel('Angle (rad)')
                ax4.legend()
                ax4.grid(True)

        # 夹爪数据
        left_gripper_data = self.synchronized_data.get('/left_tool_status', [])
        right_gripper_data = self.synchronized_data.get('/right_tool_status', [])

        if left_gripper_data and right_gripper_data:
            # 夹爪开合度
            ax5 = fig.add_subplot(gs[2, :])
            left_gripper_values = [data[0] if data is not None and len(data) > 0 else 0 for data in left_gripper_data]
            right_gripper_values = [data[0] if data is not None and len(data) > 0 else 0 for data in right_gripper_data]
            
            ax5.plot(left_gripper_values, label='Left Gripper', linewidth=2)
            ax5.plot(right_gripper_values, label='Right Gripper', linewidth=2)
            ax5.set_title('Gripper Opening')
            ax5.set_xlabel('Frame Index')
            ax5.set_ylabel('Opening (0-1)')
            ax5.legend()
            ax5.grid(True)

        plt.tight_layout()
        
        # 保存图像
        plot_path = self.output_dir / "arm_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  曲线图已保存: {plot_path}")



    def convert_separated(self):
        """⭐ 优化版本：单次读取文件，分别处理视频、action、state，再拼接（适用于大文件）"""
        start_time = datetime.now()
        
        try:
            # 1. 加载topic配置（统一自动从MCAP探测）
            self.discover_topic_configs_from_mcap()
            
            # 2. 优化：单次读取文件，同时完成数据收集和文件分析
            
            (video_data, video_timestamps, action_data, state_data, 
             video_frame_count, file_start_time, file_end_time, duration, topic_counts) = self._read_and_separate_data()
            scan_start_time = start_time
            
            # 3. 分离处理：先处理视频
            
            # 关键修复：基于最小帧数摄像头的时间戳对齐所有视频
            video_info = self._create_videos_from_data(video_data, video_timestamps, video_frame_count)
            del video_data
            gc.collect()
            
            # 4. 分离处理：处理action数据

            action_data_interpolated = self._interpolate_action_data(action_data, video_frame_count, file_start_time, file_end_time)
            del action_data
            gc.collect()
            
            # 5. 分离处理：处理state数据
            
            state_data_interpolated = self._interpolate_state_data(state_data, video_frame_count, file_start_time, file_end_time)
            del state_data
            gc.collect()
            
            # 7. 拼接数据
            
            df = self._merge_separated_data(action_data_interpolated, state_data_interpolated, video_frame_count)
            
            # 8. 评估数据质量（需要设置synchronized_data）
            # 为质量评估创建临时的synchronized_data（在删除之前保存）
            self.synchronized_data = {}
            # 合并action和state数据用于质量评估
            for topic_name, data in action_data_interpolated.items():
                if topic_name not in self.synchronized_data:
                    self.synchronized_data[topic_name] = data
            for topic_name, data in state_data_interpolated.items():
                if topic_name not in self.synchronized_data:
                    self.synchronized_data[topic_name] = data
            
            # 现在可以安全地释放临时数据
            del action_data_interpolated
            del state_data_interpolated
            gc.collect()
            
            self.evaluate_data_quality()
            
            # 9. 生成元数据文件
            self.generate_meta_files(df, video_info)
            
            # 10. 绘制曲线图（可选）
            if not getattr(self, 'no_plot', False):
                self.plot_arm_curves()
            

            
            end_time = datetime.now()
            processing_time = (end_time - scan_start_time).total_seconds()
            
            print(f"\n转换完成! 耗时: {processing_time:.2f} 秒")
            print(f"输出目录: {self.output_dir}")
            print(f"生成帧数: {len(df)}")
            print(f"整体质量评分: {self.overall_quality_score:.3f}")
            
            return ConversionReport(
                total_topics=len(self.quality_metrics),
                converted_frames=len(df),
                processing_time=processing_time,
                quality_metrics=self.quality_metrics,
                overall_quality_score=self.overall_quality_score,
                conversion_issues=self.conversion_issues,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"\n转换失败: {e}")
            raise
    
    def _read_and_separate_data(self):
        """⭐ 优化版本：单次读取文件，同时完成数据收集和文件分析"""
        print("\n⭐ 单次读取文件，同时完成数据收集和文件分析...")
        
        camera_topics = {
            "/camera_head/color/image_raw/compressed": "camera_head_rgb",
            "/camera_left/color/image_raw/compressed": "camera_left_wrist_rgb",
            "/camera_right/color/image_raw/compressed": "camera_right_wrist_rgb"
        }
        
        action_topics = [
            '/leader_left/joint_states',
            '/leader_right/joint_states',
            '/left_tool_status',
            '/right_tool_status',
            '/leader_lift/lift_up_state',
            '/leader_lift/lift_down_state'
        ]
        
        state_topics = [
            '/left_arm_controller/joint_states',
            '/right_arm_controller/joint_states',
            '/left_tool_status',
            '/right_tool_status',
            '/left_arm_controller/rm_driver/gripper_pos',
            '/right_arm_controller/rm_driver/gripper_pos',
            '/leader_lift/lift_up_state',
            '/leader_lift/lift_down_state',
            '/right_arm_controller/rm_driver/udp_lift_pos',
            '/left_arm_controller/rm_driver/udp_six_force',
            '/right_arm_controller/rm_driver/udp_six_force',
            '/left_arm_controller/rm_driver/udp_joint_speed',
            '/right_arm_controller/rm_driver/udp_joint_speed',
            '/left_arm_controller/rm_driver/udp_arm_position',
            '/right_arm_controller/rm_driver/udp_arm_position'
        ]
        
        # 初始化数据容器
        video_data = {topic: [] for topic in camera_topics.keys()}
        action_data = {topic: [] for topic in action_topics}
        state_data = {topic: [] for topic in state_topics}
        
        video_timestamps = {topic: [] for topic in camera_topics.keys()}
        action_timestamps = {topic: [] for topic in action_topics}
        state_timestamps = {topic: [] for topic in state_topics}
        
        # ⭐ 新增：用于文件分析的变量
        topic_counts = defaultdict(int)
        file_start_time = None
        file_end_time = None
        
        valid_topics = set(list(camera_topics.keys()) + action_topics + state_topics)
        message_count = 0
        total_message_count = 0  # 所有消息（包括过滤的）
        
        try:
            with open(self.mcap_file_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[mcap_ros2_decoder()])
                
                # 记录每个摄像头topic最近一次使用的header时间戳，用于检测非单调
                last_header_ts_per_camera = {}
                header_ts_fallback_to_logtime_counts = {t: 0 for t in camera_topics.keys()}
                
                for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                    total_message_count += 1
                    topic_name = channel.topic
                    
                    # 提取时间戳（优先header.stamp），对相机topic做单调性检测，必要时回退到log_time
                    ts_header = self.extract_timestamp_from_header(ros_msg)
                    ts_log = message.log_time * 1e-9
                    timestamp = ts_header if ts_header is not None else ts_log
                    
                    if topic_name in camera_topics:
                        # 对相机topic：若header时间戳非单调或重复，则回退到log_time，避免对齐全指同一帧
                        last_ts = last_header_ts_per_camera.get(topic_name, None)
                        if ts_header is None or (last_ts is not None and ts_header <= last_ts):
                            timestamp = ts_log
                            header_ts_fallback_to_logtime_counts[topic_name] += 1
                        else:
                            last_header_ts_per_camera[topic_name] = ts_header
                    
                    # ⭐ 记录文件时间范围（用于所有topic，不仅是valid_topics）
                    if topic_name in self.topic_configs:
                        topic_counts[topic_name] += 1
                        if file_start_time is None:
                            file_start_time = timestamp
                            self.start_timestamp = timestamp
                        file_end_time = timestamp
                    
                    # 只处理我们需要的topics
                    if topic_name not in valid_topics:
                        continue
                    
                    message_count += 1
                    
                    # 设置全局起始时间戳
                    if not hasattr(self, 'start_timestamp'):
                        self.start_timestamp = timestamp
                    
                    relative_timestamp = timestamp - self.start_timestamp
                    
                    # ⭐ 检查是否超过最大处理时长
                    if self.max_duration > 0 and relative_timestamp > self.max_duration:
                        print(f"  达到最大处理时长 {self.max_duration}s，停止读取")
                        break
                    
                    # 根据topic类型分类存储
                    if topic_name in camera_topics:
                        # 视频数据：立即解码
                        decoded_img = self._decode_compressed_image(ros_msg)
                        video_data[topic_name].append(decoded_img)
                        video_timestamps[topic_name].append(relative_timestamp)
                    elif topic_name in action_topics:
                        # Action数据：保存原始消息
                        action_data[topic_name].append(ros_msg)
                        action_timestamps[topic_name].append(relative_timestamp)
                    elif topic_name in state_topics:
                        # State数据：保存原始消息
                        state_data[topic_name].append(ros_msg)
                        state_timestamps[topic_name].append(relative_timestamp)
                    
                    if total_message_count % 100000 == 0:
                        video_count = sum(len(v) for v in video_data.values())
                        action_count = sum(len(v) for v in action_data.values())
                        state_count = sum(len(v) for v in state_data.values())
                        print(f"  已处理 {total_message_count} 条消息 | 视频: {video_count} 帧 | Action: {action_count} 条 | State: {state_count} 条...")
        
        except Exception as e:
            raise RuntimeError(f"读取并分离数据失败: {e}")
        
        # ⭐ 计算文件元信息
        if file_start_time is None or file_end_time is None:
            raise ValueError("没有找到有效的时间戳数据")
        
        duration = file_end_time - file_start_time
        
        # 如果相机topic存在较多header->log_time回退，打印提示
        try:
            for cam_topic, cnt in header_ts_fallback_to_logtime_counts.items():
                if cnt > 0:
                    cam_name = camera_topics.get(cam_topic, cam_topic)
                    print(f"  提示: 相机 {cam_name} 存在 {cnt} 次 header.stamp 非单调/重复，已回退使用 log_time 对齐，避免静止帧")
        except Exception:
            pass
        
        # 如果设置了最大处理时长，限制实际处理时长
        if self.max_duration > 0 and duration > self.max_duration:
            duration = self.max_duration
            file_end_time = file_start_time + self.max_duration
        
        # 计算视频帧数
        frame_counts = [len(video_data[topic]) for topic in camera_topics.keys() if topic in video_data]
        video_frame_count = min(frame_counts) if frame_counts else 0
        
        print(f"\n  ✅ 读取完成:")
        print(f"    文件时长: {duration:.2f} 秒")
        print(f"    时间范围: {file_start_time:.3f}s - {file_end_time:.3f}s")
        print(f"    相对时间: 0.000s - {duration:.3f}s")
        print(f"    总消息数: {total_message_count}")
        print(f"    相关topics: {len(topic_counts)}")
        print(f"    视频帧数: {video_frame_count}")
        print(f"    Action数据: {sum(len(v) for v in action_data.values())} 条")
        print(f"    State数据: {sum(len(v) for v in state_data.values())} 条")
        
        # 为action和state添加时间戳信息
        action_data_with_ts = (action_data, action_timestamps)
        state_data_with_ts = (state_data, state_timestamps)
        
        # ⭐ 返回数据和文件元信息
        return (video_data, video_timestamps, action_data_with_ts, state_data_with_ts, 
                video_frame_count, file_start_time, file_end_time, duration, topic_counts)
    
    def _create_videos_from_data(self, video_data, video_timestamps, video_frame_count=None):
        """从已解码的图像数据创建视频，基于时间戳对齐到最小帧数摄像头，含编码器回退与自动AV1转码"""
        print("\n创建视频文件（基于时间戳对齐）...")
        
        camera_mapping = {
            "/camera_head/color/image_raw/compressed": "camera_head_rgb",
            "/camera_left/color/image_raw/compressed": "camera_left_wrist_rgb",
            "/camera_right/color/image_raw/compressed": "camera_right_wrist_rgb"
        }
        
        # 1. 找到最小帧数的摄像头作为基准
        frame_counts = {topic: len(video_data[topic]) for topic in camera_mapping.keys() if topic in video_data}
        if not frame_counts:
            print("  错误: 没有找到视频数据")
            return {}
        
        # 找到最小帧数的摄像头
        min_frame_topic = min(frame_counts.items(), key=lambda x: x[1])[0]
        min_frame_count = frame_counts[min_frame_topic]
        reference_camera = camera_mapping[min_frame_topic]
        
        print(f"  基准摄像头: {reference_camera} ({min_frame_count} 帧)")
        
        # 获取基准摄像头的时间戳序列
        reference_timestamps = np.array(video_timestamps[min_frame_topic])
        
        video_info = {}
        
        for topic_name, camera_name in camera_mapping.items():
            if topic_name not in video_data or not video_data[topic_name]:
                continue
            
            images = video_data[topic_name]
            timestamps = np.array(video_timestamps[topic_name])
            
            
            
            if topic_name == min_frame_topic:
                # 基准摄像头：直接使用
                aligned_images = images
                
            else:
                # 其他摄像头：基于时间戳对齐到基准摄像头
                original_count = len(images)
                aligned_images = []
                for ref_ts in reference_timestamps:
                    indices_right = np.searchsorted(timestamps, ref_ts, side='right')
                    indices_left = np.searchsorted(timestamps, ref_ts, side='left')
                    idx = indices_left if indices_left < indices_right else indices_right
                    idx = np.clip(idx, 0, len(images) - 1)
                    aligned_images.append(images[idx])
                
            
            # 确保对齐后的帧数等于基准帧数
            if len(aligned_images) != video_frame_count:
                if len(aligned_images) > video_frame_count:
                    aligned_images = aligned_images[:video_frame_count]
                else:
                    if aligned_images:
                        last_frame = aligned_images[-1]
                        aligned_images = aligned_images + [last_frame] * (video_frame_count - len(aligned_images))
                    else:
                        print(f"    {camera_name}: 没有图像数据，跳过")
                        continue
            
            feature_key = f"observation.images.{camera_name}"
            camera_dir = self.videos_dir / "chunk-000" / feature_key
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = camera_dir / "episode_000000.mp4"
            
            valid_images = [img for img in aligned_images if img is not None]
            if not valid_images:
                print(f"    {camera_name}: 没有有效图像，跳过")
                continue
            
            first_img = valid_images[0]
            if not isinstance(first_img, np.ndarray):
                print(f"    {camera_name}: 图像格式错误，跳过")
                continue
            height, width = first_img.shape[:2]
            
            # 初始化视频写入器（含编码器回退）
            out, codec_used, actual_path = self._init_video_writer(video_path, width, height, TARGET_FREQUENCY)
            if out is None:
                print(f"    {camera_name}: 视频写入器初始化失败，跳过")
                continue
            
            blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            frame_count = 0
            valid_frame_count = 0
            
            for img in aligned_images:
                if img is None:
                    out.write(blank_frame)
                    frame_count += 1
                else:
                    try:
                        if not isinstance(img, np.ndarray):
                            out.write(blank_frame)
                            frame_count += 1
                            continue
                        # 调整尺寸
                        if img.shape[0] != height or img.shape[1] != width:
                            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                        # 统一dtype
                        if img.dtype != np.uint8:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                        # 通道处理
                        if len(img.shape) == 2:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif len(img.shape) == 3:
                            if img.shape[2] == 3:
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            elif img.shape[2] == 4:
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                            else:
                                img_bgr = blank_frame
                        else:
                            img_bgr = blank_frame
                        out.write(img_bgr)
                        frame_count += 1
                        valid_frame_count += 1
                    except Exception:
                        out.write(blank_frame)
                        frame_count += 1
            
            out.release()
            
            # 自动转码为 AV1 MP4（如已是MP4则覆盖原文件名；AVI则生成同名MP4）
            transcoded_path = self._transcode_to_av1_mp4(actual_path, fps=TARGET_FREQUENCY)
            final_actual_path = transcoded_path if transcoded_path else actual_path
            
            final_frame_count = video_frame_count if video_frame_count is not None else frame_count
            rel_path = os.path.relpath(str(final_actual_path), str(self.output_dir))
            video_info[camera_name] = {
                'path': rel_path,
                'frames': final_frame_count,
                'resolution': [width, height],
                'fps': TARGET_FREQUENCY
            }
            
            print(f"    {camera_name}: 保存了 {final_frame_count} 帧视频 (有效帧: {valid_frame_count}, 编码器: {codec_used}) -> {final_actual_path}")
        
        return video_info
    
    def _interpolate_action_data(self, action_data_with_ts, frame_count, start_time, end_time):
        """插值action数据到30Hz"""

        action_data, action_timestamps = action_data_with_ts
        # ⭐ 修复：使用与统一处理模式相同的时间戳生成方式
        # 统一处理模式使用: np.arange(max_frames) / TARGET_FREQUENCY
        # 确保时间戳范围一致：[0, (frame_count-1)/TARGET_FREQUENCY]
        target_timestamps = np.arange(frame_count) / TARGET_FREQUENCY
        
        action_data_interpolated = {}
        for topic_name in action_data.keys():
            if action_data[topic_name]:

                interpolated = self._interpolate_batch_data(
                    action_data[topic_name],
                    np.array(action_timestamps[topic_name]),
                    target_timestamps
                )
                action_data_interpolated[topic_name] = interpolated
        

        return action_data_interpolated
    
    def _interpolate_state_data(self, state_data_with_ts, frame_count, start_time, end_time):
        """插值state数据到30Hz"""

        state_data, state_timestamps = state_data_with_ts
        # ⭐ 修复：使用与统一处理模式相同的时间戳生成方式
        # 统一处理模式使用: np.arange(max_frames) / TARGET_FREQUENCY
        # 确保时间戳范围一致：[0, (frame_count-1)/TARGET_FREQUENCY]
        target_timestamps = np.arange(frame_count) / TARGET_FREQUENCY
        
        state_data_interpolated = {}
        for topic_name in state_data.keys():
            if state_data[topic_name]:

                interpolated = self._interpolate_batch_data(
                    state_data[topic_name],
                    np.array(state_timestamps[topic_name]),
                    target_timestamps
                )
                state_data_interpolated[topic_name] = interpolated
        

        return state_data_interpolated
    
    def _merge_separated_data(self, action_data, state_data, frame_count):
        """合并分离处理的数据为LeRobot格式"""
        print("\n合并数据为LeRobot格式...")
        
        data_rows = []
        image_timestamps = np.arange(frame_count) / TARGET_FREQUENCY
        
        for i in range(frame_count):
            row = {}
            
            # 1. ACTION数据：使用主臂数据（leader）
            # 左主臂关节数据
            left_leader_arm_data = [0.0] * 7
            if '/leader_left/joint_states' in action_data and i < len(action_data['/leader_left/joint_states']):
                joint_data = action_data['/leader_left/joint_states'][i]
                if joint_data is not None:
                    if isinstance(joint_data, list):
                        left_leader_arm_data = [float(x) for x in joint_data[:7]]
                    elif isinstance(joint_data, np.ndarray):
                        left_leader_arm_data = [float(x) for x in joint_data[:7]]
                    elif hasattr(joint_data, 'position'):
                        left_leader_arm_data = [float(x) for x in joint_data.position[:7]]
            
            # 右主臂关节数据
            right_leader_arm_data = [0.0] * 7
            if '/leader_right/joint_states' in action_data and i < len(action_data['/leader_right/joint_states']):
                joint_data = action_data['/leader_right/joint_states'][i]
                if joint_data is not None:
                    if isinstance(joint_data, list):
                        right_leader_arm_data = [float(x) for x in joint_data[:7]]
                    elif isinstance(joint_data, np.ndarray):
                        right_leader_arm_data = [float(x) for x in joint_data[:7]]
                    elif hasattr(joint_data, 'position'):
                        right_leader_arm_data = [float(x) for x in joint_data.position[:7]]
            
            # 左主臂夹爪数据（用于action）
            left_leader_gripper_value = 0.0
            if '/left_tool_status' in action_data and i < len(action_data['/left_tool_status']):
                gripper_data = action_data['/left_tool_status'][i]
                if gripper_data is not None:
                    if isinstance(gripper_data, list) and len(gripper_data) > 0:
                        left_leader_gripper_value = float(gripper_data[0])
                    elif isinstance(gripper_data, np.ndarray) and len(gripper_data) > 0:
                        left_leader_gripper_value = float(gripper_data[0])
                    elif hasattr(gripper_data, 'data'):
                        left_leader_gripper_value = float(gripper_data.data)
            
            # 右主臂夹爪数据（用于action）
            right_leader_gripper_value = 0.0
            if '/right_tool_status' in action_data and i < len(action_data['/right_tool_status']):
                gripper_data = action_data['/right_tool_status'][i]
                if gripper_data is not None:
                    if isinstance(gripper_data, list) and len(gripper_data) > 0:
                        right_leader_gripper_value = float(gripper_data[0])
                    elif isinstance(gripper_data, np.ndarray) and len(gripper_data) > 0:
                        right_leader_gripper_value = float(gripper_data[0])
                    elif hasattr(gripper_data, 'data'):
                        right_leader_gripper_value = float(gripper_data.data)
            
            # 获取Lift状态（用于action）
            lift_value = 0.0
            lift_up_value = 0.0
            if '/leader_lift/lift_up_state' in action_data and i < len(action_data['/leader_lift/lift_up_state']):
                lift_up_data = action_data['/leader_lift/lift_up_state'][i]
                if lift_up_data is not None:
                    try:
                        lift_up_value = float(lift_up_data if not hasattr(lift_up_data, '__len__') else lift_up_data[0])
                    except Exception:
                        lift_up_value = 0.0
            
            lift_down_value = 0.0
            if '/leader_lift/lift_down_state' in action_data and i < len(action_data['/leader_lift/lift_down_state']):
                lift_down_data = action_data['/leader_lift/lift_down_state'][i]
                if lift_down_data is not None:
                    try:
                        lift_down_value = float(lift_down_data if not hasattr(lift_down_data, '__len__') else lift_down_data[0])
                    except Exception:
                        lift_down_value = 0.0
            
            # 合并lift状态: 1=上升, 0=不动, -1=下降
            if lift_up_value > 0.5:
                lift_value = 1.0
            elif lift_down_value > 0.5:
                lift_value = -1.0
            else:
                lift_value = 0.0
            
            # 1. action（主臂数据：左主臂7关节 + 左主臂夹爪 + 右主臂7关节 + 右主臂夹爪 + Lift状态，共17维）
            row['action'] = np.array(left_leader_arm_data + [left_leader_gripper_value] + right_leader_arm_data + [right_leader_gripper_value] + [lift_value], dtype=np.float32)
            
            # 2. STATE数据：使用从臂数据（follower）
            # 左从臂关节数据
            left_follower_arm_data = [0.0] * 7
            if '/left_arm_controller/joint_states' in state_data and i < len(state_data['/left_arm_controller/joint_states']):
                joint_data = state_data['/left_arm_controller/joint_states'][i]
                if joint_data is not None:
                    if isinstance(joint_data, list):
                        left_follower_arm_data = [float(x) for x in joint_data[:7]]
                    elif isinstance(joint_data, np.ndarray):
                        left_follower_arm_data = [float(x) for x in joint_data[:7]]
                    elif hasattr(joint_data, 'position'):
                        left_follower_arm_data = [float(x) for x in joint_data.position[:7]]
            
            # 右从臂关节数据
            right_follower_arm_data = [0.0] * 7
            if '/right_arm_controller/joint_states' in state_data and i < len(state_data['/right_arm_controller/joint_states']):
                joint_data = state_data['/right_arm_controller/joint_states'][i]
                if joint_data is not None:
                    if isinstance(joint_data, list):
                        right_follower_arm_data = [float(x) for x in joint_data[:7]]
                    elif isinstance(joint_data, np.ndarray):
                        right_follower_arm_data = [float(x) for x in joint_data[:7]]
                    elif hasattr(joint_data, 'position'):
                        right_follower_arm_data = [float(x) for x in joint_data.position[:7]]
            
            # 夹爪数据
            left_gripper_value = 0.0
            if '/left_tool_status' in state_data and i < len(state_data['/left_tool_status']):
                gripper_data = state_data['/left_tool_status'][i]
                if gripper_data is not None:
                    if isinstance(gripper_data, list) and len(gripper_data) > 0:
                        left_gripper_value = float(gripper_data[0])
                    elif isinstance(gripper_data, np.ndarray) and len(gripper_data) > 0:
                        left_gripper_value = float(gripper_data[0])
                    elif hasattr(gripper_data, 'data'):
                        left_gripper_value = float(gripper_data.data)
            
            right_gripper_value = 0.0
            if '/right_tool_status' in state_data and i < len(state_data['/right_tool_status']):
                gripper_data = state_data['/right_tool_status'][i]
                if gripper_data is not None:
                    if isinstance(gripper_data, list) and len(gripper_data) > 0:
                        right_gripper_value = float(gripper_data[0])
                    elif isinstance(gripper_data, np.ndarray) and len(gripper_data) > 0:
                        right_gripper_value = float(gripper_data[0])
                    elif hasattr(gripper_data, 'data'):
                        right_gripper_value = float(gripper_data.data)
            
            # Lift状态
            state_lift_value = 0.0
            if '/right_arm_controller/rm_driver/udp_lift_pos' in state_data and i < len(state_data['/right_arm_controller/rm_driver/udp_lift_pos']):
                lift_data = state_data['/right_arm_controller/rm_driver/udp_lift_pos'][i]
                if lift_data is not None:
                    if hasattr(lift_data, 'height'):
                        state_lift_value = float(lift_data.height) / 10.0
                    elif isinstance(lift_data, (list, np.ndarray)) and len(lift_data) > 0:
                        state_lift_value = float(lift_data[0]) / 10.0
            
            # 获取左从动臂六维力信息 (200Hz)
            left_force_data = [0.0] * 6  # force_fx, force_fy, force_fz, force_mx, force_my, force_mz
            if '/left_arm_controller/rm_driver/udp_six_force' in state_data and i < len(state_data['/left_arm_controller/rm_driver/udp_six_force']):
                force_msg = state_data['/left_arm_controller/rm_driver/udp_six_force'][i]
                if force_msg is not None:
                    try:
                        # 提取六维力数据：force_fx, force_fy, force_fz, force_mx, force_my, force_mz
                        if hasattr(force_msg, 'force_fx'):
                            left_force_data[0] = float(force_msg.force_fx)
                        if hasattr(force_msg, 'force_fy'):
                            left_force_data[1] = float(force_msg.force_fy)
                        if hasattr(force_msg, 'force_fz'):
                            left_force_data[2] = float(force_msg.force_fz)
                        if hasattr(force_msg, 'force_mx'):
                            left_force_data[3] = float(force_msg.force_mx)
                        if hasattr(force_msg, 'force_my'):
                            left_force_data[4] = float(force_msg.force_my)
                        if hasattr(force_msg, 'force_mz'):
                            left_force_data[5] = float(force_msg.force_mz)
                    except Exception:
                        left_force_data = [0.0] * 6
            
            # 获取右从动臂六维力信息 (200Hz)
            right_force_data = [0.0] * 6  # force_fx, force_fy, force_fz, force_mx, force_my, force_mz
            if '/right_arm_controller/rm_driver/udp_six_force' in state_data and i < len(state_data['/right_arm_controller/rm_driver/udp_six_force']):
                force_msg = state_data['/right_arm_controller/rm_driver/udp_six_force'][i]
                if force_msg is not None:
                    try:
                        # 提取六维力数据：force_fx, force_fy, force_fz, force_mx, force_my, force_mz
                        if hasattr(force_msg, 'force_fx'):
                            right_force_data[0] = float(force_msg.force_fx)
                        if hasattr(force_msg, 'force_fy'):
                            right_force_data[1] = float(force_msg.force_fy)
                        if hasattr(force_msg, 'force_fz'):
                            right_force_data[2] = float(force_msg.force_fz)
                        if hasattr(force_msg, 'force_mx'):
                            right_force_data[3] = float(force_msg.force_mx)
                        if hasattr(force_msg, 'force_my'):
                            right_force_data[4] = float(force_msg.force_my)
                        if hasattr(force_msg, 'force_mz'):
                            right_force_data[5] = float(force_msg.force_mz)
                    except Exception:
                        right_force_data = [0.0] * 6
            
            # 获取左从动臂关节速度 (200Hz)
            left_velocity_data = [0.0] * 7  # 7个关节的速度
            if '/left_arm_controller/rm_driver/udp_joint_speed' in state_data and i < len(state_data['/left_arm_controller/rm_driver/udp_joint_speed']):
                velocity_msg = state_data['/left_arm_controller/rm_driver/udp_joint_speed'][i]
                if velocity_msg is not None:
                    try:
                        # 提取关节速度数据 - 优先使用joint_speed属性（Jointspeed消息类型，Float32Array (7)）
                        if hasattr(velocity_msg, 'joint_speed'):
                            joint_speed = velocity_msg.joint_speed
                            # 处理数组类型：list, np.ndarray, array.array, 或其他可迭代对象
                            if isinstance(joint_speed, (list, np.ndarray)) or hasattr(joint_speed, '__iter__'):
                                try:
                                    speeds = list(joint_speed)[:7]
                                    left_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                                except (TypeError, ValueError):
                                    left_velocity_data = [0.0] * 7
                            else:
                                # 单个值的情况
                                left_velocity_data = [float(joint_speed)] + [0.0] * 6
                        elif hasattr(velocity_msg, 'speed'):
                            if isinstance(velocity_msg.speed, (list, np.ndarray)):
                                speeds = list(velocity_msg.speed)[:7]
                                left_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                            else:
                                left_velocity_data = [float(velocity_msg.speed)] + [0.0] * 6
                        elif isinstance(velocity_msg, (list, np.ndarray)):
                            speeds = list(velocity_msg)[:7]
                            left_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                    except Exception:
                        left_velocity_data = [0.0] * 7
            
            # 获取右从动臂关节速度 (200Hz)
            right_velocity_data = [0.0] * 7  # 7个关节的速度
            if '/right_arm_controller/rm_driver/udp_joint_speed' in state_data and i < len(state_data['/right_arm_controller/rm_driver/udp_joint_speed']):
                velocity_msg = state_data['/right_arm_controller/rm_driver/udp_joint_speed'][i]
                if velocity_msg is not None:
                    try:
                        # 提取关节速度数据 - 优先使用joint_speed属性（Jointspeed消息类型，Float32Array (7)）
                        if hasattr(velocity_msg, 'joint_speed'):
                            joint_speed = velocity_msg.joint_speed
                            # 处理数组类型：list, np.ndarray, array.array, 或其他可迭代对象
                            if isinstance(joint_speed, (list, np.ndarray)) or hasattr(joint_speed, '__iter__'):
                                try:
                                    speeds = list(joint_speed)[:7]
                                    right_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                                except (TypeError, ValueError):
                                    right_velocity_data = [0.0] * 7
                            else:
                                # 单个值的情况
                                right_velocity_data = [float(joint_speed)] + [0.0] * 6
                        elif hasattr(velocity_msg, 'speed'):
                            if isinstance(velocity_msg.speed, (list, np.ndarray)):
                                speeds = list(velocity_msg.speed)[:7]
                                right_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                            else:
                                right_velocity_data = [float(velocity_msg.speed)] + [0.0] * 6
                        elif isinstance(velocity_msg, (list, np.ndarray)):
                            speeds = list(velocity_msg)[:7]
                            right_velocity_data = [float(x) for x in speeds] + [0.0] * (7 - len(speeds))
                    except Exception:
                        right_velocity_data = [0.0] * 7
            
            # 获取左从动臂末端位姿 (200Hz) - 位置(x,y,z) + 姿态四元数(w,x,y,z)，共7维
            left_pose_data = [0.0] * 7  # 位置3维(x,y,z) + 四元数4维(w,x,y,z)
            if '/left_arm_controller/rm_driver/udp_arm_position' in state_data and i < len(state_data['/left_arm_controller/rm_driver/udp_arm_position']):
                pose_msg = state_data['/left_arm_controller/rm_driver/udp_arm_position'][i]
                if pose_msg is not None:
                    try:
                        # 提取位置数据 - 优先使用pose属性（Jointposeorientation消息类型）
                        if hasattr(pose_msg, 'pose'):
                            pose = pose_msg.pose
                            # 提取位置（x, y, z）
                            if hasattr(pose, 'position'):
                                pos = pose.position
                                if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
                                    left_pose_data[0] = float(pos.x)
                                    left_pose_data[1] = float(pos.y)
                                    left_pose_data[2] = float(pos.z)
                            # 提取姿态（四元数 w, x, y, z）
                            if hasattr(pose, 'orientation'):
                                orient = pose.orientation
                                if hasattr(orient, 'x') and hasattr(orient, 'y') and hasattr(orient, 'z') and hasattr(orient, 'w'):
                                    left_pose_data[3] = float(orient.w)  # w
                                    left_pose_data[4] = float(orient.x)  # x
                                    left_pose_data[5] = float(orient.y)  # y
                                    left_pose_data[6] = float(orient.z)  # z
                        # 兼容旧格式：直接属性 x, y, z
                        elif hasattr(pose_msg, 'x') and hasattr(pose_msg, 'y') and hasattr(pose_msg, 'z'):
                            left_pose_data[0] = float(pose_msg.x)
                            left_pose_data[1] = float(pose_msg.y)
                            left_pose_data[2] = float(pose_msg.z)
                            if hasattr(pose_msg, 'qw') and hasattr(pose_msg, 'qx') and hasattr(pose_msg, 'qy') and hasattr(pose_msg, 'qz'):
                                left_pose_data[3] = float(pose_msg.qw)  # w
                                left_pose_data[4] = float(pose_msg.qx)  # x
                                left_pose_data[5] = float(pose_msg.qy)  # y
                                left_pose_data[6] = float(pose_msg.qz)  # z
                            elif hasattr(pose_msg, 'roll') and hasattr(pose_msg, 'pitch') and hasattr(pose_msg, 'yaw'):
                                # 如果只有RPY，转换为四元数
                                import math
                                roll = float(pose_msg.roll)
                                pitch = float(pose_msg.pitch)
                                yaw = float(pose_msg.yaw)
                                # RPY转四元数
                                cy = math.cos(yaw * 0.5)
                                sy = math.sin(yaw * 0.5)
                                cp = math.cos(pitch * 0.5)
                                sp = math.sin(pitch * 0.5)
                                cr = math.cos(roll * 0.5)
                                sr = math.sin(roll * 0.5)
                                left_pose_data[3] = cr * cp * cy + sr * sp * sy  # w
                                left_pose_data[4] = sr * cp * cy - cr * sp * sy  # x
                                left_pose_data[5] = cr * sp * cy + sr * cp * sy  # y
                                left_pose_data[6] = cr * cp * sy - sr * sp * cy  # z
                    except Exception as e:
                        if i == 0:
                            print(f"  ⚠️  提取左末端位姿失败: {type(pose_msg)}, 错误: {e}")
                            if hasattr(pose_msg, 'pose'):
                                print(f"      pose属性: {pose_msg.pose}")
                                if hasattr(pose_msg.pose, 'position'):
                                    print(f"      pose.position: {pose_msg.pose.position}")
                        left_pose_data = [0.0] * 7
            
            # 获取右从动臂末端位姿 (200Hz) - 位置(x,y,z) + 姿态四元数(w,x,y,z)，共7维
            right_pose_data = [0.0] * 7  # 位置3维(x,y,z) + 四元数4维(w,x,y,z)
            if '/right_arm_controller/rm_driver/udp_arm_position' in state_data and i < len(state_data['/right_arm_controller/rm_driver/udp_arm_position']):
                pose_msg = state_data['/right_arm_controller/rm_driver/udp_arm_position'][i]
                if pose_msg is not None:
                    try:
                        # 提取位置数据 - 优先使用pose属性（Jointposeorientation消息类型）
                        if hasattr(pose_msg, 'pose'):
                            pose = pose_msg.pose
                            # 提取位置（x, y, z）
                            if hasattr(pose, 'position'):
                                pos = pose.position
                                if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
                                    right_pose_data[0] = float(pos.x)
                                    right_pose_data[1] = float(pos.y)
                                    right_pose_data[2] = float(pos.z)
                            # 提取姿态（四元数 w, x, y, z）
                            if hasattr(pose, 'orientation'):
                                orient = pose.orientation
                                if hasattr(orient, 'x') and hasattr(orient, 'y') and hasattr(orient, 'z') and hasattr(orient, 'w'):
                                    right_pose_data[3] = float(orient.w)  # w
                                    right_pose_data[4] = float(orient.x)  # x
                                    right_pose_data[5] = float(orient.y)  # y
                                    right_pose_data[6] = float(orient.z)  # z
                        # 兼容旧格式：直接属性 x, y, z
                        elif hasattr(pose_msg, 'x') and hasattr(pose_msg, 'y') and hasattr(pose_msg, 'z'):
                            right_pose_data[0] = float(pose_msg.x)
                            right_pose_data[1] = float(pose_msg.y)
                            right_pose_data[2] = float(pose_msg.z)
                            if hasattr(pose_msg, 'qw') and hasattr(pose_msg, 'qx') and hasattr(pose_msg, 'qy') and hasattr(pose_msg, 'qz'):
                                right_pose_data[3] = float(pose_msg.qw)  # w
                                right_pose_data[4] = float(pose_msg.qx)  # x
                                right_pose_data[5] = float(pose_msg.qy)  # y
                                right_pose_data[6] = float(pose_msg.qz)  # z
                            elif hasattr(pose_msg, 'roll') and hasattr(pose_msg, 'pitch') and hasattr(pose_msg, 'yaw'):
                                # 如果只有RPY，转换为四元数
                                import math
                                roll = float(pose_msg.roll)
                                pitch = float(pose_msg.pitch)
                                yaw = float(pose_msg.yaw)
                                # RPY转四元数
                                cy = math.cos(yaw * 0.5)
                                sy = math.sin(yaw * 0.5)
                                cp = math.cos(pitch * 0.5)
                                sp = math.sin(pitch * 0.5)
                                cr = math.cos(roll * 0.5)
                                sr = math.sin(roll * 0.5)
                                right_pose_data[3] = cr * cp * cy + sr * sp * sy  # w
                                right_pose_data[4] = sr * cp * cy - cr * sp * sy  # x
                                right_pose_data[5] = cr * sp * cy + sr * cp * sy  # y
                                right_pose_data[6] = cr * cp * sy - sr * sp * cy  # z
                    except Exception as e:
                        if i == 0:
                            print(f"  ⚠️  提取右末端位姿失败: {type(pose_msg)}, 错误: {e}")
                            if hasattr(pose_msg, 'pose'):
                                print(f"      pose属性: {pose_msg.pose}")
                        right_pose_data = [0.0] * 7
            
            # 2. observation.state（从臂数据：左从臂7关节 + 左从臂夹爪 + 右从臂7关节 + 右从臂夹爪 + Lift状态 + 左六维力 + 右六维力 + 左关节速度7 + 右关节速度7 + 左末端位姿7 + 右末端位姿7，共57维）
            state_data_list = (left_follower_arm_data + [left_gripper_value] + 
                             right_follower_arm_data + [right_gripper_value] + 
                             [state_lift_value] + left_force_data + right_force_data +
                             left_velocity_data + right_velocity_data +
                             left_pose_data + right_pose_data)
            row['observation.state'] = np.array(state_data_list, dtype=np.float32)
            
            # 3. 其他字段
            row['timestamp'] = image_timestamps[i]
            row['frame_index'] = i
            row['episode_index'] = 0
            row['index'] = i
            row['task_index'] = 0
            
            data_rows.append(row)
        
        # 转换为DataFrame
        df = pd.DataFrame(data_rows)
        
        # 保存Parquet文件
        data_chunk_dir = self.data_dir / "chunk-000"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_chunk_dir / "episode_000000.parquet"
        df.to_parquet(parquet_path, index=False)
        
        print(f"  保存数据文件: {parquet_path}")
        print(f"  数据行数: {len(df)}")
        print(f"  数据列数: {len(df.columns)}")
        
        return df

    def convert(self):
        """执行完整的转换流程"""
        print("开始LeRobot v2.1标准格式转换（Linux服务器优化-统一30Hz）...")
        start_time = datetime.now()

        try:
            # 1. 加载topic配置（统一自动从MCAP探测）
            self.discover_topic_configs_from_mcap()

            # 2. 分析MCAP文件
            file_start_time, file_end_time, duration, topic_counts = self.analyze_mcap_file_for_duration()

            # 3. 分批处理数据
            self.process_mcap_data_in_batches(file_start_time, file_end_time)

            # 4. 评估数据质量
            self.evaluate_data_quality()

            # 5. 生成LeRobot v2.1标准数据集
            df, video_info = self.generate_lerobot_v21_dataset()

            # 6. 生成元数据文件
            self.generate_meta_files(df, video_info)

            # 7. 绘制曲线图
            self.plot_arm_curves()

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            print(f"\n转换完成! 耗时: {processing_time:.2f} 秒")
            print(f"输出目录: {self.output_dir}")
            print(f"生成帧数: {len(df)}")
            print(f"整体质量评分: {self.overall_quality_score:.3f}")
            print(f"统一采样频率: {TARGET_FREQUENCY} Hz")
            print(f"格式版本: LeRobot v2.1 标准")

            return ConversionReport(
                total_topics=len(self.quality_metrics),
                converted_frames=len(df),
                processing_time=processing_time,
                quality_metrics=self.quality_metrics,
                overall_quality_score=self.overall_quality_score,
                conversion_issues=self.conversion_issues,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"\n转换失败: {e}")
            raise

    def _generate_keys_mapping(self, columns):
        """生成keys映射 - LeRobot v2.1标准"""
        keys = {}
        for col in columns:
            if col in ['timestamp', 'frame_index', 'episode_index']:
                continue
            elif 'images' in col:
                keys[col] = 'video'
            else:
                keys[col] = 'scalar'
        return keys

    def _generate_features_mapping(self, columns):
        """生成features映射 - LeRobot v2.1标准"""
        features = {}
        for col in columns:
            if col in ['timestamp', 'frame_index', 'episode_index']:
                continue
            elif 'images' in col:
                # 视频特征
                features[col] = {
                    "dtype": "uint8",
                    "shape": [3, 480, 640],  # 假设图像尺寸
                    "video": True
                }
            else:
                # 标量特征
                features[col] = {
                    "dtype": "float32",
                    "shape": []
                }
        return features


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MCAP到LeRobot v2.1标准格式转换器 (Linux服务器优化版-统一30Hz)')
    parser.add_argument('--input', required=True, help='输入MCAP文件或目录路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    # Excel 配置相关参数已移除
    parser.add_argument('--max-duration', type=int, default=0, help='最大处理时长（秒），0表示不限制，处理完整文件')
    parser.add_argument('--resolution', choices=['360p', '720p'], default='720p', help='目标分辨率：360p (640x360) 或 720p (1280x720)，默认720p')
    parser.add_argument('--no-plot', action='store_true', help='不生成臂与夹爪曲线图')
    parser.add_argument('--threads', type=int, default=None, help='处理线程数（覆盖默认的半核策略）')
    parser.add_argument('--ffmpeg-threads', type=int, default=None, help='AV1转码线程数（覆盖自动CPU核心数）')
    parser.add_argument('--ffmpeg-cpu-used', type=int, default=None, help='AV1编码速度参数cpu-used（默认8，越大越快）')

    args = parser.parse_args()

    # 归一化路径：将反斜杠转为正斜杠，并解析为绝对路径
    input_path = Path(str(args.input).replace('\\', '/')).resolve()
    output_dir = Path(str(args.output).replace('\\', '/')).resolve()
    # print(f"[Paths] Input: {input_path.as_posix()}")
    # print(f"[Paths] Output: {output_dir.as_posix()}")

    # 检查输入文件
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path.as_posix()}")
        sys.exit(1)

    # 若为目录，最开始进行扫描并报告mcap数量
    mcap_files = None
    if input_path.is_dir():
        mcap_files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".mcap"]
        print(f"[Scan] 在目录中发现 {len(mcap_files)} 个MCAP文件: {input_path.as_posix()}")
        if not mcap_files:
            print(f"错误: 在目录中未找到mcap文件: {input_path.as_posix()}")
            sys.exit(1)

    # 不再支持 Excel 配置，统一从 MCAP 自动探测

    if args.threads is not None and args.threads > 0:
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
        os.environ['MKL_NUM_THREADS'] = str(args.threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.threads)
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        try:
            cv2.setNumThreads(args.threads)
        except Exception:
            pass
        print(f"[Perf] 处理线程数覆盖为: {args.threads}")

    if args.ffmpeg_threads is not None and args.ffmpeg_threads > 0:
        os.environ['FFMPEG_THREADS'] = str(args.ffmpeg_threads)
        print(f"[Perf] FFmpeg转码线程数覆盖为: {args.ffmpeg_threads}")

    if args.ffmpeg_cpu_used is not None and args.ffmpeg_cpu_used >= 0:
        os.environ['FFMPEG_CPU_USED'] = str(args.ffmpeg_cpu_used)
        print(f"[Perf] FFmpeg cpu-used 覆盖为: {args.ffmpeg_cpu_used}")

    if input_path.is_file():
        converter = MCAPToLeRobotV21StandardConverter(
            str(input_path),
            str(output_dir),
            max_duration=args.max_duration,
            resolution=args.resolution
        )
        converter.no_plot = args.no_plot
        report = converter.convert_separated()
        print("\n转换总结:")
        print(f"- 处理了 {len(report.quality_metrics)} 个topics")
        converter.validate_training_readiness()
        print(f"- 生成了 {report.converted_frames} 帧数据")
        print(f"- 整体质量评分: {report.overall_quality_score:.3f}")
        print(f"- 统一采样频率: {TARGET_FREQUENCY} Hz")
        print(f"- 格式版本: LeRobot v2.1 标准")
        if report.overall_quality_score >= 0.8:
            print("✅ 数据质量优秀，可直接用于训练!")
        elif report.overall_quality_score >= 0.6:
            print("⚠️  数据质量良好，建议检查部分topic")
        else:
            print("❌ 数据质量较差，建议重新采集")
    elif input_path.is_dir():
        print(f"发现 {len(mcap_files)} 个MCAP文件")
        for p in tqdm(mcap_files, desc="转换MCAP", unit="file"):
            rel = p.relative_to(input_path)
            sub_output = output_dir / rel.parent / p.stem
            sub_output.mkdir(parents=True, exist_ok=True)
            # print(f"\n[批量转换] {p.as_posix()} -> {sub_output.as_posix()}")
            converter = MCAPToLeRobotV21StandardConverter(
                str(p),
                str(sub_output),
                max_duration=args.max_duration,
                resolution=args.resolution
            )
            converter.no_plot = args.no_plot
            print("使用分离处理模式（先分别处理视频、action、state，再拼接）...")
            try:
                report = converter.convert_separated()
                print("\n转换总结:")
                print(f"- 处理了 {len(report.quality_metrics)} 个topics")
                converter.validate_training_readiness()
                print(f"- 生成了 {report.converted_frames} 帧数据")
                print(f"- 整体质量评分: {report.overall_quality_score:.3f}")
                print(f"- 统一采样频率: {TARGET_FREQUENCY} Hz")
                print(f"- 格式版本: LeRobot v2.1 标准")
                if report.overall_quality_score >= 0.8:
                    print("✅ 数据质量优秀，可直接用于训练!")
                elif report.overall_quality_score >= 0.6:
                    print("⚠️  数据质量良好，建议检查部分topic")
                else:
                    print("❌ 数据质量较差，建议重新采集")
            except Exception as e:
                print(f"文件转换失败: {p} -> {e}")
                continue
        print("\n批量转换完成")
    else:
        print(f"错误: 无效的输入路径: {input_path.as_posix()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
