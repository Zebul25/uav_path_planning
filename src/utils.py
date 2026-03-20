"""
工具函数
========

提供配置加载、随机种子设置、目录创建等通用工具函数.
"""

import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def set_seed(seed: int):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dirs(config: Dict[str, Any]) -> Dict[str, str]:
    """
    创建实验所需目录
    
    Args:
        config: 配置字典
        
    Returns:
        目录路径字典
    """
    logging_config = config.get('logging', {})
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dirs = {
        'log_dir': logging_config.get('log_dir', 'results/logs'),
        'model_dir': logging_config.get('model_dir', 'results/models'),
        'figure_dir': logging_config.get('figure_dir', 'results/figures'),
    }
    
    # 添加时间戳子目录
    for key in dirs:
        dirs[key] = os.path.join(dirs[key], timestamp)
        os.makedirs(dirs[key], exist_ok=True)
    
    return dirs


def get_device(device_str: str = "auto") -> torch.device:
    """
    获取计算设备
    
    Args:
        device_str: 设备字符串 ("auto", "cpu", "cuda")
        
    Returns:
        torch.device
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def linear_schedule(initial_value: float):
    """
    创建线性衰减调度器
    
    Args:
        initial_value: 初始值
        
    Returns:
        调度函数
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def exponential_schedule(initial_value: float, decay_rate: float = 0.99):
    """
    创建指数衰减调度器
    
    Args:
        initial_value: 初始值
        decay_rate: 衰减率
        
    Returns:
        调度函数
    """
    def schedule(progress_remaining: float) -> float:
        return initial_value * (decay_rate ** (1 - progress_remaining))
    return schedule


class MetricsLogger:
    """
    指标记录器
    
    用于记录和统计训练/评估指标.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        初始化记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        self.metrics: Dict[str, list] = {}
        self.episode_metrics: Dict[str, list] = {}
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def log(self, key: str, value: float):
        """记录单个值"""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def log_episode(self, metrics: Dict[str, float]):
        """记录一个episode的指标"""
        for key, value in metrics.items():
            if key not in self.episode_metrics:
                self.episode_metrics[key] = []
            self.episode_metrics[key].append(value)
    
    def get_mean(self, key: str, last_n: Optional[int] = None) -> float:
        """获取均值"""
        if key not in self.metrics:
            return 0.0
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        return np.mean(values) if values else 0.0
    
    def get_episode_stats(self, last_n: int = 100) -> Dict[str, float]:
        """获取最近N个episode的统计"""
        stats = {}
        for key, values in self.episode_metrics.items():
            recent = values[-last_n:] if len(values) > last_n else values
            stats[f'{key}_mean'] = np.mean(recent) if recent else 0.0
            stats[f'{key}_std'] = np.std(recent) if recent else 0.0
            stats[f'{key}_min'] = np.min(recent) if recent else 0.0
            stats[f'{key}_max'] = np.max(recent) if recent else 0.0
        return stats
    
    def save(self, filename: str = "metrics.npz"):
        """保存指标到文件"""
        if self.log_dir:
            save_path = os.path.join(self.log_dir, filename)
            np.savez(
                save_path,
                metrics=self.metrics,
                episode_metrics=self.episode_metrics
            )
    
    def load(self, filepath: str):
        """从文件加载指标"""
        data = np.load(filepath, allow_pickle=True)
        self.metrics = data['metrics'].item()
        self.episode_metrics = data['episode_metrics'].item()


class EarlyStopping:
    """
    早停机制
    
    当验证指标不再改善时停止训练.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        初始化早停
        
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善量
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        检查是否应该停止
        
        Args:
            value: 当前值
            
        Returns:
            是否应该停止
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def calculate_path_metrics(trajectory: np.ndarray, goal: np.ndarray) -> Dict[str, float]:
    """
    计算路径指标
    
    Args:
        trajectory: 轨迹数组, shape (N, 3)
        goal: 目标位置
        
    Returns:
        指标字典
    """
    if len(trajectory) < 2:
        return {}
    
    # 路径长度
    path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    
    # 直线距离
    straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
    
    # 路径效率
    path_efficiency = straight_distance / (path_length + 1e-8)
    
    # 最终距离
    final_distance = np.linalg.norm(trajectory[-1] - goal)
    
    # 平均高度
    avg_height = np.mean(trajectory[:, 2])
    
    # 高度变化
    height_variation = np.std(trajectory[:, 2])
    
    # 平滑度 (二阶导数的平均值)
    if len(trajectory) >= 3:
        first_diff = np.diff(trajectory, axis=0)
        second_diff = np.diff(first_diff, axis=0)
        smoothness = np.mean(np.linalg.norm(second_diff, axis=1))
    else:
        smoothness = 0.0
    
    return {
        'path_length': path_length,
        'straight_distance': straight_distance,
        'path_efficiency': path_efficiency,
        'final_distance': final_distance,
        'avg_height': avg_height,
        'height_variation': height_variation,
        'smoothness': smoothness,
    }


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    打印配置
    
    Args:
        config: 配置字典
        indent: 缩进级别
    """
    for key, value in config.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试指标记录器
    logger = MetricsLogger()
    for i in range(100):
        logger.log('reward', np.random.randn())
        if i % 10 == 0:
            logger.log_episode({
                'episode_reward': np.random.randn() * 100,
                'episode_length': np.random.randint(100, 500),
            })
    
    print(f"Mean reward: {logger.get_mean('reward', last_n=50):.4f}")
    print(f"Episode stats: {logger.get_episode_stats(last_n=5)}")
    
    # 测试早停
    early_stop = EarlyStopping(patience=3, mode='max')
    values = [1, 2, 3, 3, 3, 3, 3]
    for v in values:
        stop = early_stop(v)
        print(f"Value: {v}, Should stop: {stop}")
    
    # 测试路径指标
    trajectory = np.random.randn(100, 3).cumsum(axis=0)
    goal = trajectory[-1] + np.array([10, 10, 0])
    metrics = calculate_path_metrics(trajectory, goal)
    print(f"Path metrics: {metrics}")
