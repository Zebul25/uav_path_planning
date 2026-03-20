# DEM环境无人机路径规划强化学习实验

## 完整代码实现文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目结构](#2-项目结构)
3. [环境配置](#3-环境配置)
4. [核心代码实现](#4-核心代码实现)
5. [训练脚本](#5-训练脚本)
6. [评估与可视化](#6-评估与可视化)
7. [使用指南](#7-使用指南)

---

## 1. 项目概述

### 1.1 实验目标

在真实DEM（数字高程模型）地形环境中，使用深度强化学习训练无人机自主规划三维路径，实现：

- 从起点安全飞行到目标点
- 避开地形障碍（山峰、建筑）
- 保持安全飞行高度
- 优化路径长度和能耗
- 对比SAC与PPO算法性能

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       系统架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐ │
│  │   DEM数据    │───▶│  地形环境    │───▶│  3D地形网格   │ │
│  │  (GeoTIFF)   │    │   构建器     │    │               │ │
│  └──────────────┘    └──────────────┘    └───────┬───────┘ │
│                                                  │         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────▼───────┐ │
│  │   RL Agent   │◀──▶│  Gymnasium   │◀──▶│   无人机      │ │
│  │  (SAC/PPO)   │    │    环境      │    │    仿真       │ │
│  └──────────────┘    └──────────────┘    └───────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 核心设计参数

| 组件 | 参数 | 说明 |
|------|------|------|
| **状态空间** | 14维连续 | 位置、速度、目标、地形 |
| **动作空间** | 3维连续 | 三轴速度指令 |
| **奖励函数** | 多目标加权 | 距离+安全+能耗+时间 |
| **地形尺寸** | 1000m×1000m | 可配置 |
| **仿真步长** | 0.1秒 | 可配置 |

---

## 2. 项目结构

```
uav_path_planning/
├── README.md                 # 项目说明
├── DOCUMENTATION.md          # 完整技术文档
├── requirements.txt          # 依赖包列表
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── __init__.py          # 包初始化
│   ├── dem_loader.py        # DEM数据加载器
│   ├── uav_model.py         # 无人机动力学模型
│   ├── uav_env.py           # Gymnasium环境实现
│   └── utils.py             # 工具函数
├── train/
│   ├── train_sac.py         # SAC训练脚本
│   ├── train_ppo.py         # PPO训练脚本
│   └── evaluate.py          # 模型评估脚本
├── visualization/
│   └── visualizer.py        # 3D可视化工具
├── data/
│   └── dem/                 # DEM数据目录
└── results/
    ├── models/              # 训练好的模型
    ├── logs/                # 训练日志
    └── figures/             # 可视化图片
```

---

## 3. 环境配置

### 3.1 requirements.txt

```txt
# Deep Learning Framework
torch>=1.12.0
torchvision>=0.13.0

# Reinforcement Learning
stable-baselines3>=2.0.0
gymnasium>=0.29.0
shimmy>=1.0.0

# Scientific Computing
numpy>=1.21.0
scipy>=1.7.0

# DEM Processing
rasterio>=1.3.0
pyproj>=3.4.0

# Visualization
matplotlib>=3.5.0
plotly>=5.10.0

# Configuration
pyyaml>=6.0
argparse

# Logging
tensorboard>=2.10.0
tqdm>=4.64.0

# Utilities
pandas>=1.4.0
```

### 3.2 config/config.yaml

```yaml
# DEM无人机路径规划实验配置文件

# ==================== 环境配置 ====================
environment:
  # 地形参数
  terrain:
    size_x: 1000.0          # 地形X方向尺寸 (米)
    size_y: 1000.0          # 地形Y方向尺寸 (米)
    resolution: 10.0        # 网格分辨率 (米)
    max_height: 500.0       # 最大高程 (米)
    dem_file: null          # DEM文件路径，null则使用生成地形
    terrain_type: "hills"   # 地形类型: flat, hills, mountains, valley
  
  # 无人机参数
  uav:
    max_velocity_xy: 15.0   # 最大水平速度 (m/s)
    max_velocity_z: 5.0     # 最大垂直速度 (m/s)
    max_turn_rate: 45.0     # 最大转向率 (度/s)
    sensor_range: 100.0     # 感知范围 (米)
    collision_radius: 2.0   # 碰撞半径 (米)
  
  # 任务参数
  task:
    safe_altitude: 20.0     # 最小安全高度 (米)
    max_altitude: 300.0     # 最大飞行高度 (米)
    goal_threshold: 10.0    # 到达目标阈值 (米)
    max_steps: 500          # 最大步数
    dt: 0.1                 # 仿真时间步 (秒)
  
  # 起点和终点设置
  spawn:
    start_random: true      # 随机起点
    start_position: [50.0, 50.0, 100.0]   # 固定起点
    goal_random: true       # 随机终点
    goal_position: [950.0, 950.0, 100.0]  # 固定终点
    min_start_goal_dist: 500.0  # 起点终点最小距离

# ==================== 奖励配置 ====================
reward:
  # 奖励权重
  distance_weight: 10.0     # 距离奖励权重
  goal_reward: 500.0        # 到达目标奖励
  collision_penalty: -200.0 # 碰撞惩罚
  low_altitude_penalty: -2.0  # 低空惩罚系数
  high_altitude_penalty: -0.5 # 高空惩罚系数
  energy_penalty: -0.01     # 能耗惩罚系数
  time_penalty: -0.1        # 时间惩罚
  
  # 奖励裁剪
  clip_reward: true
  reward_min: -10.0
  reward_max: 10.0

# ==================== 训练配置 ====================
training:
  # 通用参数
  seed: 42
  device: "auto"            # auto, cpu, cuda
  total_timesteps: 1000000  # 总训练步数
  eval_freq: 10000          # 评估频率
  n_eval_episodes: 10       # 评估episode数
  save_freq: 50000          # 模型保存频率
  
  # SAC参数
  sac:
    learning_rate: 0.0003
    buffer_size: 1000000
    batch_size: 256
    gamma: 0.99
    tau: 0.005
    ent_coef: "auto"
    target_entropy: "auto"
    train_freq: 1
    gradient_steps: 1
    learning_starts: 10000
    policy_kwargs:
      net_arch: [256, 256]
  
  # PPO参数
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    clip_range_vf: null
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
    n_envs: 16
    policy_kwargs:
      net_arch:
        pi: [256, 256]
        vf: [256, 256]

# ==================== 评估配置 ====================
evaluation:
  n_episodes: 100
  render: false
  save_trajectories: true
  metrics:
    - success_rate
    - avg_path_length
    - avg_episode_length
    - collision_rate
    - avg_cumulative_reward

# ==================== 可视化配置 ====================
visualization:
  render_mode: "3d"         # 2d, 3d
  save_animation: true
  animation_fps: 30
  show_terrain: true
  show_trajectory: true
  show_goal: true
  colormap: "terrain"

# ==================== 日志配置 ====================
logging:
  log_dir: "results/logs"
  model_dir: "results/models"
  figure_dir: "results/figures"
  tensorboard: true
  verbose: 1
```

---

## 4. 核心代码实现

### 4.1 src/\_\_init\_\_.py

```python
"""
DEM环境无人机路径规划强化学习实验
====================================

模块说明:
- dem_loader: DEM数据加载和处理
- uav_env: Gymnasium环境实现
- uav_model: 无人机动力学模型
- utils: 工具函数
"""

from .dem_loader import DEMLoader, TerrainGenerator
from .uav_env import UAVPathPlanningEnv
from .uav_model import UAVModel
from .utils import load_config, set_seed, create_dirs

__version__ = "1.0.0"
__author__ = "UAV Research Team"

__all__ = [
    "DEMLoader",
    "TerrainGenerator", 
    "UAVPathPlanningEnv",
    "UAVModel",
    "load_config",
    "set_seed",
    "create_dirs",
]
```

### 4.2 src/dem_loader.py

```python
"""
DEM数据加载器和地形生成器
========================

功能:
1. 从GeoTIFF文件加载真实DEM数据
2. 生成模拟地形用于测试
3. 地形数据预处理和归一化
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


class DEMLoader:
    """
    DEM数据加载器
    
    支持从GeoTIFF文件加载真实DEM数据,
    并提供地形高度查询接口.
    """
    
    def __init__(
        self,
        dem_file: Optional[str] = None,
        size_x: float = 1000.0,
        size_y: float = 1000.0,
        resolution: float = 10.0,
        max_height: float = 500.0
    ):
        """
        初始化DEM加载器
        
        Args:
            dem_file: DEM文件路径 (GeoTIFF格式)
            size_x: 地形X方向尺寸 (米)
            size_y: 地形Y方向尺寸 (米)
            resolution: 网格分辨率 (米)
            max_height: 最大高程 (米)
        """
        self.dem_file = dem_file
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.max_height = max_height
        
        # 计算网格尺寸
        self.grid_x = int(size_x / resolution)
        self.grid_y = int(size_y / resolution)
        
        # 加载或生成地形
        if dem_file is not None:
            self.elevation_data = self._load_from_file(dem_file)
        else:
            self.elevation_data = None
            
        # 创建插值器
        self._create_interpolator()
        
    def _load_from_file(self, dem_file: str) -> np.ndarray:
        """
        从GeoTIFF文件加载DEM数据
        
        Args:
            dem_file: DEM文件路径
            
        Returns:
            高程数据数组
        """
        try:
            import rasterio
            
            with rasterio.open(dem_file) as src:
                elevation = src.read(1)
                
                # 重采样到目标分辨率
                from scipy.ndimage import zoom
                
                target_shape = (self.grid_y, self.grid_x)
                zoom_factors = (
                    target_shape[0] / elevation.shape[0],
                    target_shape[1] / elevation.shape[1]
                )
                elevation = zoom(elevation, zoom_factors, order=1)
                
                # 处理无效值
                elevation = np.nan_to_num(elevation, nan=0.0)
                
                # 归一化到指定高度范围
                elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
                elevation = elevation * self.max_height
                
                return elevation
                
        except ImportError:
            print("警告: rasterio未安装，使用生成地形")
            return None
        except Exception as e:
            print(f"警告: 加载DEM文件失败 ({e})，使用生成地形")
            return None
    
    def _create_interpolator(self):
        """创建地形高度插值器"""
        if self.elevation_data is not None:
            x = np.linspace(0, self.size_x, self.grid_x)
            y = np.linspace(0, self.size_y, self.grid_y)
            self.interpolator = RegularGridInterpolator(
                (y, x), 
                self.elevation_data,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
        else:
            self.interpolator = None
    
    def get_elevation(self, x: float, y: float) -> float:
        """
        获取指定位置的地形高度
        
        Args:
            x: X坐标 (米)
            y: Y坐标 (米)
            
        Returns:
            地形高度 (米)
        """
        if self.interpolator is not None:
            return float(self.interpolator([[y, x]])[0])
        return 0.0
    
    def get_elevation_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        批量获取地形高度
        
        Args:
            positions: 位置数组, shape (N, 2) - [[x, y], ...]
            
        Returns:
            高度数组, shape (N,)
        """
        if self.interpolator is not None:
            # 注意: RegularGridInterpolator 需要 (y, x) 顺序
            points = positions[:, [1, 0]]
            return self.interpolator(points)
        return np.zeros(len(positions))
    
    def get_elevation_grid(self) -> np.ndarray:
        """
        获取完整的地形网格数据
        
        Returns:
            高程网格, shape (grid_y, grid_x)
        """
        if self.elevation_data is not None:
            return self.elevation_data.copy()
        return np.zeros((self.grid_y, self.grid_x))
    
    def set_elevation_data(self, data: np.ndarray):
        """
        设置地形数据
        
        Args:
            data: 高程数据数组
        """
        self.elevation_data = data
        self._create_interpolator()


class TerrainGenerator:
    """
    程序化地形生成器
    
    支持生成多种类型的模拟地形用于测试和训练.
    """
    
    def __init__(
        self,
        size_x: float = 1000.0,
        size_y: float = 1000.0,
        resolution: float = 10.0,
        max_height: float = 500.0,
        seed: Optional[int] = None
    ):
        """
        初始化地形生成器
        
        Args:
            size_x: 地形X方向尺寸 (米)
            size_y: 地形Y方向尺寸 (米)
            resolution: 网格分辨率 (米)
            max_height: 最大高程 (米)
            seed: 随机种子
        """
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.max_height = max_height
        
        self.grid_x = int(size_x / resolution)
        self.grid_y = int(size_y / resolution)
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate(self, terrain_type: str = "hills") -> DEMLoader:
        """
        生成地形
        
        Args:
            terrain_type: 地形类型
                - "flat": 平坦地形
                - "hills": 丘陵地形
                - "mountains": 山区地形
                - "valley": 峡谷地形
                - "mixed": 混合地形
                
        Returns:
            DEMLoader实例
        """
        generators = {
            "flat": self._generate_flat,
            "hills": self._generate_hills,
            "mountains": self._generate_mountains,
            "valley": self._generate_valley,
            "mixed": self._generate_mixed,
        }
        
        if terrain_type not in generators:
            raise ValueError(f"未知地形类型: {terrain_type}")
        
        elevation = generators[terrain_type]()
        
        # 创建DEMLoader并设置数据
        loader = DEMLoader(
            dem_file=None,
            size_x=self.size_x,
            size_y=self.size_y,
            resolution=self.resolution,
            max_height=self.max_height
        )
        loader.set_elevation_data(elevation)
        
        return loader
    
    def _generate_flat(self) -> np.ndarray:
        """生成平坦地形"""
        base_height = self.max_height * 0.1
        noise = np.random.randn(self.grid_y, self.grid_x) * 5
        noise = gaussian_filter(noise, sigma=3)
        return np.ones((self.grid_y, self.grid_x)) * base_height + noise
    
    def _generate_hills(self) -> np.ndarray:
        """生成丘陵地形"""
        elevation = np.zeros((self.grid_y, self.grid_x))
        
        # 添加多个高斯山丘
        n_hills = np.random.randint(5, 15)
        for _ in range(n_hills):
            cx = np.random.uniform(0, self.grid_x)
            cy = np.random.uniform(0, self.grid_y)
            height = np.random.uniform(0.3, 0.8) * self.max_height
            sigma = np.random.uniform(10, 30)
            
            x = np.arange(self.grid_x)
            y = np.arange(self.grid_y)
            xx, yy = np.meshgrid(x, y)
            
            hill = height * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            elevation += hill
        
        # 添加噪声
        noise = np.random.randn(self.grid_y, self.grid_x) * 10
        noise = gaussian_filter(noise, sigma=2)
        elevation += noise
        
        # 裁剪到有效范围
        elevation = np.clip(elevation, 0, self.max_height)
        
        return elevation
    
    def _generate_mountains(self) -> np.ndarray:
        """生成山区地形 (使用Diamond-Square算法)"""
        # 确保尺寸是2的幂+1
        size = max(self.grid_x, self.grid_y)
        n = int(np.ceil(np.log2(size - 1)))
        ds_size = 2**n + 1
        
        elevation = self._diamond_square(ds_size, roughness=1.0)
        
        # 裁剪到目标尺寸
        elevation = elevation[:self.grid_y, :self.grid_x]
        
        # 归一化
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
        elevation = elevation * self.max_height
        
        return elevation
    
    def _diamond_square(self, size: int, roughness: float = 1.0) -> np.ndarray:
        """Diamond-Square算法生成分形地形"""
        elevation = np.zeros((size, size))
        
        # 初始化四角
        elevation[0, 0] = np.random.random()
        elevation[0, size-1] = np.random.random()
        elevation[size-1, 0] = np.random.random()
        elevation[size-1, size-1] = np.random.random()
        
        step = size - 1
        scale = roughness
        
        while step > 1:
            half = step // 2
            
            # Diamond步骤
            for y in range(half, size - 1, step):
                for x in range(half, size - 1, step):
                    avg = (
                        elevation[y - half, x - half] +
                        elevation[y - half, x + half] +
                        elevation[y + half, x - half] +
                        elevation[y + half, x + half]
                    ) / 4.0
                    elevation[y, x] = avg + (np.random.random() - 0.5) * scale
            
            # Square步骤
            for y in range(0, size, half):
                for x in range((y + half) % step, size, step):
                    count = 0
                    total = 0.0
                    if y >= half:
                        total += elevation[y - half, x]
                        count += 1
                    if y + half < size:
                        total += elevation[y + half, x]
                        count += 1
                    if x >= half:
                        total += elevation[y, x - half]
                        count += 1
                    if x + half < size:
                        total += elevation[y, x + half]
                        count += 1
                    elevation[y, x] = total / count + (np.random.random() - 0.5) * scale
            
            step = half
            scale *= 0.5
        
        return elevation
    
    def _generate_valley(self) -> np.ndarray:
        """生成峡谷地形"""
        x = np.linspace(0, 1, self.grid_x)
        y = np.linspace(0, 1, self.grid_y)
        xx, yy = np.meshgrid(x, y)
        
        # 创建V形峡谷
        valley_center = 0.5 + 0.2 * np.sin(yy * np.pi * 2)
        valley_depth = np.abs(xx - valley_center)
        
        elevation = valley_depth * self.max_height
        
        # 添加山脊
        ridge_left = np.exp(-((xx - valley_center + 0.3)**2) / 0.02) * self.max_height * 0.5
        ridge_right = np.exp(-((xx - valley_center - 0.3)**2) / 0.02) * self.max_height * 0.5
        elevation += ridge_left + ridge_right
        
        # 添加噪声
        noise = np.random.randn(self.grid_y, self.grid_x) * 15
        noise = gaussian_filter(noise, sigma=3)
        elevation += noise
        
        elevation = np.clip(elevation, 0, self.max_height)
        
        return elevation
    
    def _generate_mixed(self) -> np.ndarray:
        """生成混合地形"""
        # 组合多种地形
        hills = self._generate_hills()
        mountains = self._generate_mountains()
        
        # 创建混合mask
        x = np.linspace(0, 1, self.grid_x)
        y = np.linspace(0, 1, self.grid_y)
        xx, yy = np.meshgrid(x, y)
        mask = (np.sin(xx * np.pi) * np.sin(yy * np.pi) + 1) / 2
        
        elevation = hills * mask + mountains * (1 - mask)
        
        return elevation


def create_terrain(config: Dict[str, Any]) -> DEMLoader:
    """
    根据配置创建地形
    
    Args:
        config: 配置字典
        
    Returns:
        DEMLoader实例
    """
    terrain_config = config['environment']['terrain']
    
    dem_file = terrain_config.get('dem_file')
    
    if dem_file is not None:
        # 从文件加载
        loader = DEMLoader(
            dem_file=dem_file,
            size_x=terrain_config['size_x'],
            size_y=terrain_config['size_y'],
            resolution=terrain_config['resolution'],
            max_height=terrain_config['max_height']
        )
    else:
        # 生成地形
        generator = TerrainGenerator(
            size_x=terrain_config['size_x'],
            size_y=terrain_config['size_y'],
            resolution=terrain_config['resolution'],
            max_height=terrain_config['max_height'],
            seed=config['training'].get('seed', 42)
        )
        loader = generator.generate(terrain_config.get('terrain_type', 'hills'))
    
    return loader
```

### 4.3 src/uav_model.py

```python
"""
无人机动力学模型
================

实现简化的无人机动力学模型,包括:
1. 位置和速度状态
2. 速度控制输入
3. 碰撞检测
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class UAVState:
    """无人机状态"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx, vy, vz]
    
    def copy(self) -> 'UAVState':
        """创建状态副本"""
        return UAVState(
            position=self.position.copy(),
            velocity=self.velocity.copy()
        )


class UAVModel:
    """
    无人机动力学模型
    
    采用简化的速度控制模型:
    - 输入为速度指令
    - 一阶惯性响应
    - 速度和加速度限制
    """
    
    def __init__(
        self,
        max_velocity_xy: float = 15.0,
        max_velocity_z: float = 5.0,
        max_acceleration: float = 5.0,
        time_constant: float = 0.3,
        collision_radius: float = 2.0,
        dt: float = 0.1
    ):
        """
        初始化无人机模型
        
        Args:
            max_velocity_xy: 最大水平速度 (m/s)
            max_velocity_z: 最大垂直速度 (m/s)
            max_acceleration: 最大加速度 (m/s^2)
            time_constant: 速度响应时间常数 (s)
            collision_radius: 碰撞检测半径 (m)
            dt: 仿真时间步 (s)
        """
        self.max_velocity_xy = max_velocity_xy
        self.max_velocity_z = max_velocity_z
        self.max_acceleration = max_acceleration
        self.time_constant = time_constant
        self.collision_radius = collision_radius
        self.dt = dt
        
        # 状态
        self.state = UAVState()
        
        # 速度指令
        self.velocity_cmd = np.zeros(3)
        
    def reset(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None
    ) -> UAVState:
        """
        重置无人机状态
        
        Args:
            position: 初始位置 [x, y, z]
            velocity: 初始速度 [vx, vy, vz]
            
        Returns:
            重置后的状态
        """
        self.state.position = position if position is not None else np.zeros(3)
        self.state.velocity = velocity if velocity is not None else np.zeros(3)
        self.velocity_cmd = np.zeros(3)
        
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> UAVState:
        """
        执行一步仿真
        
        Args:
            action: 速度指令 [vx_cmd, vy_cmd, vz_cmd], 范围 [-1, 1]
            
        Returns:
            更新后的状态
        """
        # 将动作映射到速度指令
        self.velocity_cmd = np.array([
            action[0] * self.max_velocity_xy,
            action[1] * self.max_velocity_xy,
            action[2] * self.max_velocity_z
        ])
        
        # 一阶惯性响应
        alpha = self.dt / (self.time_constant + self.dt)
        velocity_new = (1 - alpha) * self.state.velocity + alpha * self.velocity_cmd
        
        # 限制加速度
        acceleration = (velocity_new - self.state.velocity) / self.dt
        acc_magnitude = np.linalg.norm(acceleration)
        if acc_magnitude > self.max_acceleration:
            acceleration = acceleration / acc_magnitude * self.max_acceleration
            velocity_new = self.state.velocity + acceleration * self.dt
        
        # 限制速度
        velocity_new[0] = np.clip(velocity_new[0], -self.max_velocity_xy, self.max_velocity_xy)
        velocity_new[1] = np.clip(velocity_new[1], -self.max_velocity_xy, self.max_velocity_xy)
        velocity_new[2] = np.clip(velocity_new[2], -self.max_velocity_z, self.max_velocity_z)
        
        # 更新位置
        position_new = self.state.position + velocity_new * self.dt
        
        # 更新状态
        self.state.velocity = velocity_new
        self.state.position = position_new
        
        return self.state.copy()
    
    def get_state(self) -> UAVState:
        """获取当前状态"""
        return self.state.copy()
    
    def get_position(self) -> np.ndarray:
        """获取当前位置"""
        return self.state.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """获取当前速度"""
        return self.state.velocity.copy()
    
    def check_collision(
        self,
        terrain_height: float,
        safe_altitude: float = 20.0
    ) -> bool:
        """
        检查是否与地形碰撞
        
        Args:
            terrain_height: 当前位置地形高度 (m)
            safe_altitude: 安全高度裕度 (m)
            
        Returns:
            是否碰撞
        """
        height_above_ground = self.state.position[2] - terrain_height
        return height_above_ground < self.collision_radius
    
    def get_height_above_ground(self, terrain_height: float) -> float:
        """
        获取距地面高度
        
        Args:
            terrain_height: 当前位置地形高度 (m)
            
        Returns:
            距地面高度 (m)
        """
        return self.state.position[2] - terrain_height
    
    def check_boundary(
        self,
        bounds_x: Tuple[float, float],
        bounds_y: Tuple[float, float],
        bounds_z: Tuple[float, float]
    ) -> bool:
        """
        检查是否越界
        
        Args:
            bounds_x: X方向边界 (min, max)
            bounds_y: Y方向边界 (min, max)
            bounds_z: Z方向边界 (min, max)
            
        Returns:
            是否越界
        """
        x, y, z = self.state.position
        return (
            x < bounds_x[0] or x > bounds_x[1] or
            y < bounds_y[0] or y > bounds_y[1] or
            z < bounds_z[0] or z > bounds_z[1]
        )
    
    def get_info(self) -> Dict[str, Any]:
        """获取状态信息字典"""
        return {
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'speed': float(np.linalg.norm(self.state.velocity)),
            'velocity_cmd': self.velocity_cmd.tolist(),
        }
```

### 4.4 src/uav_env.py

```python
"""
无人机路径规划Gymnasium环境
===========================

实现标准Gymnasium接口的无人机路径规划环境,
支持在DEM地形上进行强化学习训练.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .dem_loader import DEMLoader, TerrainGenerator, create_terrain
from .uav_model import UAVModel


class UAVPathPlanningEnv(gym.Env):
    """
    无人机路径规划环境
    
    观测空间 (14维):
        - 无人机位置 (3): x, y, z (归一化)
        - 无人机速度 (3): vx, vy, vz (归一化)
        - 相对目标位置 (3): dx, dy, dz (归一化)
        - 到目标距离 (1): 归一化
        - 周围地形高度 (4): 前/左/右/下方 (归一化)
    
    动作空间 (3维连续):
        - 前向速度指令: [-1, 1]
        - 侧向速度指令: [-1, 1]
        - 垂直速度指令: [-1, 1]
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None
    ):
        """
        初始化环境
        
        Args:
            config: 配置字典
            render_mode: 渲染模式
        """
        super().__init__()
        
        # 加载配置
        self.config = config or self._default_config()
        self._parse_config()
        
        # 渲染设置
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        
        # 初始化地形
        self._init_terrain()
        
        # 初始化无人机模型
        self.uav = UAVModel(
            max_velocity_xy=self.max_velocity_xy,
            max_velocity_z=self.max_velocity_z,
            collision_radius=self.collision_radius,
            dt=self.dt
        )
        
        # 定义动作空间和观测空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # 状态变量
        self.goal_position = np.zeros(3)
        self.current_step = 0
        self.trajectory: List[np.ndarray] = []
        self.prev_distance = 0.0
        
        # 统计信息
        self.episode_reward = 0.0
        self.episode_length = 0
        self.collision_count = 0
        self.success_count = 0
        
    def _default_config(self) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            'environment': {
                'terrain': {
                    'size_x': 1000.0,
                    'size_y': 1000.0,
                    'resolution': 10.0,
                    'max_height': 500.0,
                    'dem_file': None,
                    'terrain_type': 'hills',
                },
                'uav': {
                    'max_velocity_xy': 15.0,
                    'max_velocity_z': 5.0,
                    'collision_radius': 2.0,
                    'sensor_range': 100.0,
                },
                'task': {
                    'safe_altitude': 20.0,
                    'max_altitude': 300.0,
                    'goal_threshold': 10.0,
                    'max_steps': 500,
                    'dt': 0.1,
                },
                'spawn': {
                    'start_random': True,
                    'start_position': [50.0, 50.0, 100.0],
                    'goal_random': True,
                    'goal_position': [950.0, 950.0, 100.0],
                    'min_start_goal_dist': 500.0,
                },
            },
            'reward': {
                'distance_weight': 10.0,
                'goal_reward': 500.0,
                'collision_penalty': -200.0,
                'low_altitude_penalty': -2.0,
                'high_altitude_penalty': -0.5,
                'energy_penalty': -0.01,
                'time_penalty': -0.1,
                'clip_reward': True,
                'reward_min': -10.0,
                'reward_max': 10.0,
            },
            'training': {
                'seed': 42,
            }
        }
    
    def _parse_config(self):
        """解析配置"""
        env_config = self.config['environment']
        terrain_config = env_config['terrain']
        uav_config = env_config['uav']
        task_config = env_config['task']
        spawn_config = env_config['spawn']
        reward_config = self.config['reward']
        
        # 地形参数
        self.terrain_size_x = terrain_config['size_x']
        self.terrain_size_y = terrain_config['size_y']
        self.terrain_resolution = terrain_config['resolution']
        self.terrain_max_height = terrain_config['max_height']
        self.dem_file = terrain_config.get('dem_file')
        self.terrain_type = terrain_config.get('terrain_type', 'hills')
        
        # 无人机参数
        self.max_velocity_xy = uav_config['max_velocity_xy']
        self.max_velocity_z = uav_config['max_velocity_z']
        self.collision_radius = uav_config['collision_radius']
        self.sensor_range = uav_config['sensor_range']
        
        # 任务参数
        self.safe_altitude = task_config['safe_altitude']
        self.max_altitude = task_config['max_altitude']
        self.goal_threshold = task_config['goal_threshold']
        self.max_steps = task_config['max_steps']
        self.dt = task_config['dt']
        
        # 生成参数
        self.start_random = spawn_config['start_random']
        self.start_position = np.array(spawn_config['start_position'])
        self.goal_random = spawn_config['goal_random']
        self.goal_position_default = np.array(spawn_config['goal_position'])
        self.min_start_goal_dist = spawn_config['min_start_goal_dist']
        
        # 奖励参数
        self.distance_weight = reward_config['distance_weight']
        self.goal_reward = reward_config['goal_reward']
        self.collision_penalty = reward_config['collision_penalty']
        self.low_altitude_penalty = reward_config['low_altitude_penalty']
        self.high_altitude_penalty = reward_config['high_altitude_penalty']
        self.energy_penalty = reward_config['energy_penalty']
        self.time_penalty = reward_config['time_penalty']
        self.clip_reward = reward_config['clip_reward']
        self.reward_min = reward_config['reward_min']
        self.reward_max = reward_config['reward_max']
        
    def _init_terrain(self):
        """初始化地形"""
        if self.dem_file is not None:
            self.terrain = DEMLoader(
                dem_file=self.dem_file,
                size_x=self.terrain_size_x,
                size_y=self.terrain_size_y,
                resolution=self.terrain_resolution,
                max_height=self.terrain_max_height
            )
        else:
            generator = TerrainGenerator(
                size_x=self.terrain_size_x,
                size_y=self.terrain_size_y,
                resolution=self.terrain_resolution,
                max_height=self.terrain_max_height,
                seed=self.config['training'].get('seed', 42)
            )
            self.terrain = generator.generate(self.terrain_type)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 生成起点
        if self.start_random:
            start_pos = self._random_position()
        else:
            start_pos = self.start_position.copy()
        
        # 确保起点高度安全
        terrain_height = self.terrain.get_elevation(start_pos[0], start_pos[1])
        start_pos[2] = max(start_pos[2], terrain_height + self.safe_altitude + 20)
        
        # 生成终点
        if self.goal_random:
            self.goal_position = self._random_goal(start_pos)
        else:
            self.goal_position = self.goal_position_default.copy()
        
        # 确保终点高度合适
        goal_terrain = self.terrain.get_elevation(
            self.goal_position[0], self.goal_position[1]
        )
        self.goal_position[2] = max(
            self.goal_position[2], 
            goal_terrain + self.safe_altitude
        )
        
        # 重置无人机
        self.uav.reset(position=start_pos)
        
        # 重置状态
        self.current_step = 0
        self.trajectory = [start_pos.copy()]
        self.prev_distance = self._distance_to_goal()
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # 获取观测
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作 [vx_cmd, vy_cmd, vz_cmd], 范围 [-1, 1]
            
        Returns:
            observation: 观测
            reward: 奖励
            terminated: 是否终止 (到达目标或碰撞)
            truncated: 是否截断 (超时)
            info: 额外信息
        """
        # 执行动作
        action = np.clip(action, -1.0, 1.0)
        self.uav.step(action)
        
        # 更新状态
        self.current_step += 1
        position = self.uav.get_position()
        self.trajectory.append(position.copy())
        
        # 获取地形高度
        terrain_height = self.terrain.get_elevation(position[0], position[1])
        height_above_ground = position[2] - terrain_height
        
        # 检查终止条件
        terminated = False
        truncated = False
        
        # 检查碰撞
        collision = self.uav.check_collision(terrain_height)
        if collision:
            terminated = True
            self.collision_count += 1
        
        # 检查是否到达目标
        distance = self._distance_to_goal()
        reached_goal = distance < self.goal_threshold
        if reached_goal:
            terminated = True
            self.success_count += 1
        
        # 检查边界
        out_of_bounds = self.uav.check_boundary(
            bounds_x=(0, self.terrain_size_x),
            bounds_y=(0, self.terrain_size_y),
            bounds_z=(0, self.max_altitude + 100)
        )
        if out_of_bounds:
            terminated = True
        
        # 检查超时
        if self.current_step >= self.max_steps:
            truncated = True
        
        # 计算奖励
        reward = self._compute_reward(
            action=action,
            collision=collision,
            reached_goal=reached_goal,
            height_above_ground=height_above_ground,
            distance=distance
        )
        
        # 更新状态
        self.prev_distance = distance
        self.episode_reward += reward
        self.episode_length = self.current_step
        
        # 获取观测
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取观测向量"""
        position = self.uav.get_position()
        velocity = self.uav.get_velocity()
        
        # 归一化位置
        pos_norm = np.array([
            position[0] / self.terrain_size_x,
            position[1] / self.terrain_size_y,
            position[2] / self.max_altitude
        ])
        
        # 归一化速度
        vel_norm = np.array([
            velocity[0] / self.max_velocity_xy,
            velocity[1] / self.max_velocity_xy,
            velocity[2] / self.max_velocity_z
        ])
        
        # 相对目标位置
        relative_goal = self.goal_position - position
        rel_goal_norm = np.array([
            relative_goal[0] / self.terrain_size_x,
            relative_goal[1] / self.terrain_size_y,
            relative_goal[2] / self.max_altitude
        ])
        
        # 到目标距离
        distance = self._distance_to_goal()
        max_dist = np.sqrt(self.terrain_size_x**2 + self.terrain_size_y**2)
        dist_norm = distance / max_dist
        
        # 周围地形高度
        terrain_heights = self._get_terrain_around(position)
        terrain_norm = terrain_heights / self.terrain_max_height
        
        # 组合观测
        observation = np.concatenate([
            pos_norm,           # 3
            vel_norm,           # 3
            rel_goal_norm,      # 3
            [dist_norm],        # 1
            terrain_norm,       # 4
        ]).astype(np.float32)
        
        return observation
    
    def _get_terrain_around(self, position: np.ndarray) -> np.ndarray:
        """
        获取周围地形高度
        
        Args:
            position: 当前位置
            
        Returns:
            [前方, 左方, 右方, 当前位置] 的地形高度
        """
        x, y, z = position
        delta = self.sensor_range / 2
        
        # 采样点
        points = np.array([
            [x + delta, y],      # 前方
            [x, y + delta],      # 左方
            [x, y - delta],      # 右方
            [x, y],              # 当前
        ])
        
        # 裁剪到边界内
        points[:, 0] = np.clip(points[:, 0], 0, self.terrain_size_x - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.terrain_size_y - 1)
        
        # 获取高度
        heights = self.terrain.get_elevation_batch(points)
        
        return heights
    
    def _compute_reward(
        self,
        action: np.ndarray,
        collision: bool,
        reached_goal: bool,
        height_above_ground: float,
        distance: float
    ) -> float:
        """
        计算奖励
        
        Args:
            action: 执行的动作
            collision: 是否碰撞
            reached_goal: 是否到达目标
            height_above_ground: 距地面高度
            distance: 到目标距离
            
        Returns:
            奖励值
        """
        reward = 0.0
        
        # 1. 距离奖励 (接近目标获得正奖励)
        distance_reward = (self.prev_distance - distance) * self.distance_weight
        reward += distance_reward
        
        # 2. 到达目标奖励
        if reached_goal:
            reward += self.goal_reward
        
        # 3. 碰撞惩罚
        if collision:
            reward += self.collision_penalty
        
        # 4. 高度安全奖励
        if height_above_ground < self.safe_altitude:
            penalty = (self.safe_altitude - height_above_ground) * self.low_altitude_penalty
            reward += penalty
        elif height_above_ground > self.max_altitude:
            penalty = (height_above_ground - self.max_altitude) * self.high_altitude_penalty
            reward += penalty
        
        # 5. 能耗惩罚
        energy = np.linalg.norm(action)
        reward += energy * self.energy_penalty
        
        # 6. 时间惩罚
        reward += self.time_penalty
        
        # 奖励裁剪
        if self.clip_reward:
            reward = np.clip(reward, self.reward_min, self.reward_max)
        
        return float(reward)
    
    def _distance_to_goal(self) -> float:
        """计算到目标的距离"""
        position = self.uav.get_position()
        return float(np.linalg.norm(position - self.goal_position))
    
    def _random_position(self) -> np.ndarray:
        """生成随机位置"""
        margin = 50.0
        x = np.random.uniform(margin, self.terrain_size_x - margin)
        y = np.random.uniform(margin, self.terrain_size_y - margin)
        
        terrain_height = self.terrain.get_elevation(x, y)
        z = terrain_height + np.random.uniform(
            self.safe_altitude + 20, 
            self.max_altitude * 0.5
        )
        
        return np.array([x, y, z])
    
    def _random_goal(self, start_pos: np.ndarray) -> np.ndarray:
        """生成随机目标位置,确保与起点有足够距离"""
        for _ in range(100):
            goal = self._random_position()
            dist = np.linalg.norm(goal[:2] - start_pos[:2])
            if dist >= self.min_start_goal_dist:
                return goal
        
        # 如果找不到合适的位置,使用默认位置
        return self.goal_position_default.copy()
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        position = self.uav.get_position()
        terrain_height = self.terrain.get_elevation(position[0], position[1])
        
        return {
            'position': position.tolist(),
            'velocity': self.uav.get_velocity().tolist(),
            'goal_position': self.goal_position.tolist(),
            'distance_to_goal': self._distance_to_goal(),
            'height_above_ground': position[2] - terrain_height,
            'step': self.current_step,
            'episode_reward': self.episode_reward,
        }
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            self._render_3d()
        elif self.render_mode == 'rgb_array':
            return self._render_to_array()
    
    def _render_3d(self):
        """3D渲染"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
        
        self.ax.clear()
        
        # 绘制地形
        terrain_grid = self.terrain.get_elevation_grid()
        x = np.linspace(0, self.terrain_size_x, terrain_grid.shape[1])
        y = np.linspace(0, self.terrain_size_y, terrain_grid.shape[0])
        X, Y = np.meshgrid(x, y)
        
        self.ax.plot_surface(
            X, Y, terrain_grid,
            cmap='terrain',
            alpha=0.6,
            antialiased=True
        )
        
        # 绘制轨迹
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                'b-', linewidth=2, label='Trajectory'
            )
        
        # 绘制无人机位置
        pos = self.uav.get_position()
        self.ax.scatter(
            pos[0], pos[1], pos[2],
            c='blue', s=100, marker='o', label='UAV'
        )
        
        # 绘制目标
        self.ax.scatter(
            self.goal_position[0],
            self.goal_position[1],
            self.goal_position[2],
            c='red', s=200, marker='*', label='Goal'
        )
        
        # 设置标签
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(f'Step: {self.current_step}, Reward: {self.episode_reward:.2f}')
        self.ax.legend()
        
        plt.draw()
        plt.pause(0.01)
    
    def _render_to_array(self) -> np.ndarray:
        """渲染为图像数组"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self._render_3d()
        
        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        
        return data
    
    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def get_trajectory(self) -> np.ndarray:
        """获取当前episode的轨迹"""
        return np.array(self.trajectory)


def make_env(config: Dict[str, Any], rank: int = 0, seed: int = 0):
    """
    创建环境的工厂函数,用于向量化环境
    
    Args:
        config: 配置字典
        rank: 环境索引
        seed: 随机种子
    """
    def _init():
        env = UAVPathPlanningEnv(config=config)
        env.reset(seed=seed + rank)
        return env
    return _init
```

### 4.5 src/utils.py

```python
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
```

---

## 5. 训练脚本

### 5.1 train/train_sac.py

```python
"""
SAC算法训练脚本
===============

使用Soft Actor-Critic算法训练无人机路径规划.
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.uav_env import UAVPathPlanningEnv
from src.utils import (
    load_config,
    save_config,
    set_seed,
    create_dirs,
    get_device,
    print_config
)


def create_env(config, seed=0):
    """创建训练环境"""
    env = UAVPathPlanningEnv(config=config)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_sac(config_path: str, resume: str = None):
    """
    训练SAC模型
    
    Args:
        config_path: 配置文件路径
        resume: 恢复训练的模型路径
    """
    # 加载配置
    config = load_config(config_path)
    training_config = config['training']
    sac_config = training_config['sac']
    
    # 设置随机种子
    seed = training_config.get('seed', 42)
    set_seed(seed)
    
    # 创建目录
    dirs = create_dirs(config)
    print(f"\n{'='*60}")
    print(f"SAC Training for UAV Path Planning")
    print(f"{'='*60}")
    print(f"Log directory: {dirs['log_dir']}")
    print(f"Model directory: {dirs['model_dir']}")
    
    # 保存配置
    save_config(config, os.path.join(dirs['log_dir'], 'config.yaml'))
    
    # 获取设备
    device = get_device(training_config.get('device', 'auto'))
    print(f"\nUsing device: {device}")
    
    # 创建环境
    print("\nCreating environment...")
    env = DummyVecEnv([lambda: create_env(config, seed)])
    
    # 创建评估环境
    eval_env = DummyVecEnv([lambda: create_env(config, seed + 1000)])
    
    # 创建SAC模型
    print("\nCreating SAC model...")
    policy_kwargs = dict(
        net_arch=sac_config['policy_kwargs']['net_arch']
    )
    
    if resume:
        print(f"Resuming from: {resume}")
        model = SAC.load(resume, env=env, device=device)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=sac_config['learning_rate'],
            buffer_size=sac_config['buffer_size'],
            batch_size=sac_config['batch_size'],
            gamma=sac_config['gamma'],
            tau=sac_config['tau'],
            ent_coef=sac_config['ent_coef'],
            target_entropy=sac_config['target_entropy'],
            train_freq=sac_config['train_freq'],
            gradient_steps=sac_config['gradient_steps'],
            learning_starts=sac_config['learning_starts'],
            policy_kwargs=policy_kwargs,
            verbose=training_config.get('verbose', 1),
            device=device,
            tensorboard_log=dirs['log_dir'],
            seed=seed,
        )
    
    # 创建回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dirs['model_dir'],
        log_path=dirs['log_dir'],
        eval_freq=training_config['eval_freq'],
        n_eval_episodes=training_config['n_eval_episodes'],
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config['save_freq'],
        save_path=dirs['model_dir'],
        name_prefix='sac_checkpoint',
        verbose=1,
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # 开始训练
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"Total timesteps: {training_config['total_timesteps']}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    # 保存最终模型
    final_model_path = os.path.join(dirs['model_dir'], 'sac_final.zip')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # 打印训练统计
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Training time: {training_time}")
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return model, dirs


def main():
    parser = argparse.ArgumentParser(description='Train SAC for UAV Path Planning')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to model to resume training'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # 开始训练
    train_sac(args.config, args.resume)


if __name__ == "__main__":
    main()
```

### 5.2 train/train_ppo.py

```python
"""
PPO算法训练脚本
===============

使用Proximal Policy Optimization算法训练无人机路径规划.
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.uav_env import UAVPathPlanningEnv, make_env
from src.utils import (
    load_config,
    save_config,
    set_seed,
    create_dirs,
    get_device,
    print_config
)


def create_parallel_envs(config, n_envs, seed=0):
    """创建并行环境"""
    env_fns = [make_env(config, rank=i, seed=seed) for i in range(n_envs)]
    
    if n_envs > 1:
        try:
            env = SubprocVecEnv(env_fns)
        except Exception as e:
            print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
            env = DummyVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    
    return env


def train_ppo(config_path: str, resume: str = None):
    """
    训练PPO模型
    
    Args:
        config_path: 配置文件路径
        resume: 恢复训练的模型路径
    """
    # 加载配置
    config = load_config(config_path)
    training_config = config['training']
    ppo_config = training_config['ppo']
    
    # 设置随机种子
    seed = training_config.get('seed', 42)
    set_seed(seed)
    
    # 创建目录
    dirs = create_dirs(config)
    print(f"\n{'='*60}")
    print(f"PPO Training for UAV Path Planning")
    print(f"{'='*60}")
    print(f"Log directory: {dirs['log_dir']}")
    print(f"Model directory: {dirs['model_dir']}")
    
    # 保存配置
    save_config(config, os.path.join(dirs['log_dir'], 'config.yaml'))
    
    # 获取设备
    device = get_device(training_config.get('device', 'auto'))
    print(f"\nUsing device: {device}")
    
    # 创建并行环境
    n_envs = ppo_config.get('n_envs', 8)
    print(f"\nCreating {n_envs} parallel environments...")
    env = create_parallel_envs(config, n_envs, seed)
    
    # 创建评估环境 (单个环境)
    eval_env = DummyVecEnv([make_env(config, rank=0, seed=seed + 1000)])
    
    # 创建PPO模型
    print("\nCreating PPO model...")
    policy_kwargs = dict(
        net_arch=ppo_config['policy_kwargs']['net_arch']
    )
    
    if resume:
        print(f"Resuming from: {resume}")
        model = PPO.load(resume, env=env, device=device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=ppo_config['learning_rate'],
            n_steps=ppo_config['n_steps'],
            batch_size=ppo_config['batch_size'],
            n_epochs=ppo_config['n_epochs'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_range=ppo_config['clip_range'],
            clip_range_vf=ppo_config['clip_range_vf'],
            ent_coef=ppo_config['ent_coef'],
            vf_coef=ppo_config['vf_coef'],
            max_grad_norm=ppo_config['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            verbose=training_config.get('verbose', 1),
            device=device,
            tensorboard_log=dirs['log_dir'],
            seed=seed,
        )
    
    # 计算实际的评估和保存频率
    eval_freq = max(training_config['eval_freq'] // n_envs, 1)
    save_freq = max(training_config['save_freq'] // n_envs, 1)
    
    # 创建回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dirs['model_dir'],
        log_path=dirs['log_dir'],
        eval_freq=eval_freq,
        n_eval_episodes=training_config['n_eval_episodes'],
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=dirs['model_dir'],
        name_prefix='ppo_checkpoint',
        verbose=1,
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # 开始训练
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"Total timesteps: {training_config['total_timesteps']}")
    print(f"Number of environments: {n_envs}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    # 保存最终模型
    final_model_path = os.path.join(dirs['model_dir'], 'ppo_final.zip')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # 打印训练统计
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Training time: {training_time}")
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return model, dirs


def main():
    parser = argparse.ArgumentParser(description='Train PPO for UAV Path Planning')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to model to resume training'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    train_ppo(args.config, args.resume)


if __name__ == "__main__":
    main()
```

### 5.3 train/evaluate.py

```python
"""
模型评估脚本
============

评估训练好的模型性能并生成统计报告.
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor

from src.uav_env import UAVPathPlanningEnv
from src.utils import load_config, set_seed, calculate_path_metrics


def load_model(model_path: str, env):
    """加载模型"""
    if 'sac' in model_path.lower():
        model = SAC.load(model_path, env=env)
        print(f"Loaded SAC model from {model_path}")
    elif 'ppo' in model_path.lower():
        model = PPO.load(model_path, env=env)
        print(f"Loaded PPO model from {model_path}")
    else:
        try:
            model = SAC.load(model_path, env=env)
            print(f"Loaded SAC model from {model_path}")
        except:
            model = PPO.load(model_path, env=env)
            print(f"Loaded PPO model from {model_path}")
    
    return model


def evaluate_model(
    model,
    env,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    save_trajectories: bool = True
):
    """评估模型"""
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success': [],
        'collision': [],
        'timeout': [],
        'final_distances': [],
        'path_lengths': [],
        'path_efficiencies': [],
        'trajectories': [] if save_trajectories else None,
    }
    
    print(f"\nEvaluating model for {n_episodes} episodes...")
    
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        trajectory = [info['position'].copy()]
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            trajectory.append(info['position'].copy())
            
            if render:
                env.render()
            
            done = terminated or truncated
        
        # 记录结果
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        results['final_distances'].append(info['distance_to_goal'])
        
        # 判断结果类型
        success = info['distance_to_goal'] < env.goal_threshold
        collision = info['height_above_ground'] < env.collision_radius
        timeout = episode_length >= env.max_steps
        
        results['success'].append(success)
        results['collision'].append(collision)
        results['timeout'].append(timeout)
        
        # 计算路径指标
        trajectory = np.array(trajectory)
        path_metrics = calculate_path_metrics(trajectory, env.goal_position)
        results['path_lengths'].append(path_metrics.get('path_length', 0))
        results['path_efficiencies'].append(path_metrics.get('path_efficiency', 0))
        
        if save_trajectories:
            results['trajectories'].append(trajectory.tolist())
    
    return results


def compute_statistics(results: dict) -> dict:
    """计算统计指标"""
    stats = {}
    
    # 成功率
    stats['success_rate'] = np.mean(results['success']) * 100
    stats['collision_rate'] = np.mean(results['collision']) * 100
    stats['timeout_rate'] = np.mean(results['timeout']) * 100
    
    # 奖励统计
    rewards = results['episode_rewards']
    stats['reward_mean'] = np.mean(rewards)
    stats['reward_std'] = np.std(rewards)
    stats['reward_min'] = np.min(rewards)
    stats['reward_max'] = np.max(rewards)
    
    # Episode长度统计
    lengths = results['episode_lengths']
    stats['length_mean'] = np.mean(lengths)
    stats['length_std'] = np.std(lengths)
    
    # 成功episode的统计
    success_mask = np.array(results['success'])
    if np.any(success_mask):
        success_rewards = np.array(rewards)[success_mask]
        success_lengths = np.array(lengths)[success_mask]
        success_paths = np.array(results['path_lengths'])[success_mask]
        success_efficiencies = np.array(results['path_efficiencies'])[success_mask]
        
        stats['success_reward_mean'] = np.mean(success_rewards)
        stats['success_length_mean'] = np.mean(success_lengths)
        stats['success_path_length_mean'] = np.mean(success_paths)
        stats['success_efficiency_mean'] = np.mean(success_efficiencies)
    else:
        stats['success_reward_mean'] = 0
        stats['success_length_mean'] = 0
        stats['success_path_length_mean'] = 0
        stats['success_efficiency_mean'] = 0
    
    # 最终距离统计
    final_dists = results['final_distances']
    stats['final_distance_mean'] = np.mean(final_dists)
    stats['final_distance_std'] = np.std(final_dists)
    
    return stats


def print_statistics(stats: dict):
    """打印统计结果"""
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    
    print(f"\n📊 Overall Performance:")
    print(f"   Success Rate:    {stats['success_rate']:.2f}%")
    print(f"   Collision Rate:  {stats['collision_rate']:.2f}%")
    print(f"   Timeout Rate:    {stats['timeout_rate']:.2f}%")
    
    print(f"\n💰 Reward Statistics:")
    print(f"   Mean:  {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}")
    print(f"   Range: [{stats['reward_min']:.2f}, {stats['reward_max']:.2f}]")
    
    print(f"\n⏱️ Episode Length:")
    print(f"   Mean:  {stats['length_mean']:.1f} ± {stats['length_std']:.1f}")
    
    print(f"\n🎯 Success Episodes:")
    print(f"   Mean Reward:      {stats['success_reward_mean']:.2f}")
    print(f"   Mean Length:      {stats['success_length_mean']:.1f}")
    print(f"   Mean Path Length: {stats['success_path_length_mean']:.1f} m")
    print(f"   Mean Efficiency:  {stats['success_efficiency_mean']:.2f}")
    
    print(f"\n📍 Final Distance:")
    print(f"   Mean: {stats['final_distance_mean']:.2f} ± {stats['final_distance_std']:.2f} m")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-dir', type=str, default='results/evaluation')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    config = load_config(args.config)
    
    render_mode = 'human' if args.render else None
    env = UAVPathPlanningEnv(config=config, render_mode=render_mode)
    env = Monitor(env)
    
    model = load_model(args.model, env)
    
    results = evaluate_model(
        model=model,
        env=env,
        n_episodes=args.episodes,
        deterministic=True,
        render=args.render,
        save_trajectories=True
    )
    
    stats = compute_statistics(results)
    print_statistics(stats)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    stats_path = os.path.join(save_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")
    
    env.close()


if __name__ == "__main__":
    main()
```

---

## 6. 评估与可视化

### 6.1 visualization/visualizer.py

```python
"""
3D可视化工具
============

提供无人机路径规划的3D可视化.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC, PPO
from src.uav_env import UAVPathPlanningEnv
from src.utils import load_config, set_seed


class UAVVisualizer:
    """无人机路径规划可视化器"""
    
    def __init__(self, env, figsize=(12, 8), colormap='terrain'):
        self.env = env
        self.figsize = figsize
        self.colormap = colormap
        
        self.terrain_grid = env.terrain.get_elevation_grid()
        self.terrain_x = np.linspace(0, env.terrain_size_x, self.terrain_grid.shape[1])
        self.terrain_y = np.linspace(0, env.terrain_size_y, self.terrain_grid.shape[0])
        self.X, self.Y = np.meshgrid(self.terrain_x, self.terrain_y)
        
        self.fig = None
        self.ax = None
        
    def setup_figure(self):
        """设置图形"""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=30, azim=45)
        return self.fig, self.ax
    
    def render_terrain(self, alpha=0.6):
        """渲染地形"""
        surf = self.ax.plot_surface(
            self.X, self.Y, self.terrain_grid,
            cmap=self.colormap,
            alpha=alpha,
            antialiased=True,
            shade=True
        )
        return surf
    
    def render_trajectory(self, trajectory, color='blue', linewidth=2, label='Trajectory'):
        """渲染轨迹"""
        line, = self.ax.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            color=color, linewidth=linewidth, label=label
        )
        return line
    
    def visualize_episode(self, trajectory, start, goal, title="UAV Path", save_path=None):
        """可视化单个episode"""
        self.setup_figure()
        self.render_terrain()
        self.render_trajectory(trajectory)
        
        self.ax.scatter(start[0], start[1], start[2], c='green', s=150, marker='o', label='Start')
        self.ax.scatter(goal[0], goal[1], goal[2], c='red', s=200, marker='*', label='Goal')
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(title)
        self.ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def create_animation(self, trajectory, goal, fps=30, interval=50, save_path=None):
        """创建轨迹动画"""
        self.setup_figure()
        self.render_terrain(alpha=0.5)
        self.ax.scatter(goal[0], goal[1], goal[2], c='red', s=200, marker='*', label='Goal')
        
        line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        uav, = self.ax.plot([], [], [], 'bo', markersize=10)
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title("UAV Flight Animation")
        self.ax.legend()
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            uav.set_data([], [])
            uav.set_3d_properties([])
            return line, uav
        
        def update(frame):
            line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
            line.set_3d_properties(trajectory[:frame+1, 2])
            uav.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
            uav.set_3d_properties([trajectory[frame, 2]])
            return line, uav
        
        anim = FuncAnimation(
            self.fig, update, frames=len(trajectory),
            init_func=init, interval=interval, blit=True
        )
        
        if save_path:
            if save_path.endswith('.gif'):
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            else:
                anim.save(save_path, fps=fps)
            print(f"Animation saved to: {save_path}")
        
        plt.show()
        return anim


def visualize_terrain_types(save_path=None):
    """可视化不同地形类型"""
    from src.dem_loader import TerrainGenerator
    
    terrain_types = ['flat', 'hills', 'mountains', 'valley', 'mixed']
    
    fig = plt.figure(figsize=(20, 8))
    
    for idx, ttype in enumerate(terrain_types):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        generator = TerrainGenerator(seed=42)
        loader = generator.generate(ttype)
        elevation = loader.get_elevation_grid()
        
        x = np.linspace(0, loader.size_x, elevation.shape[1])
        y = np.linspace(0, loader.size_y, elevation.shape[0])
        X, Y = np.meshgrid(x, y)
        
        ax.plot_surface(X, Y, elevation, cmap='terrain', alpha=0.8)
        ax.set_title(f'Terrain: {ttype}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def run_visualization(model_path, config_path, n_episodes=1, save_animation=False, output_dir='results/figures'):
    """运行可视化"""
    config = load_config(config_path)
    set_seed(config['training'].get('seed', 42))
    
    env = UAVPathPlanningEnv(config=config)
    
    if 'sac' in model_path.lower():
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)
    
    visualizer = UAVVisualizer(env)
    os.makedirs(output_dir, exist_ok=True)
    
    for ep in range(n_episodes):
        print(f"\nEpisode {ep + 1}/{n_episodes}")
        
        obs, info = env.reset()
        trajectory = [env.uav.get_position().copy()]
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.uav.get_position().copy())
            total_reward += reward
            done = terminated or truncated
        
        trajectory = np.array(trajectory)
        
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {len(trajectory)}")
        print(f"  Final distance: {info['distance_to_goal']:.2f} m")
        
        success = info['distance_to_goal'] < env.goal_threshold
        status = "Success" if success else "Failed"
        title = f"Episode {ep + 1} - {status} (Reward: {total_reward:.1f})"
        
        save_path = os.path.join(output_dir, f'episode_{ep + 1}.png')
        visualizer.visualize_episode(
            trajectory=trajectory,
            start=trajectory[0],
            goal=env.goal_position,
            title=title,
            save_path=save_path
        )
        
        if save_animation:
            anim_path = os.path.join(output_dir, f'episode_{ep + 1}.gif')
            visualizer.create_animation(
                trajectory=trajectory,
                goal=env.goal_position,
                save_path=anim_path
            )
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize UAV Path Planning')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--animation', action='store_true')
    parser.add_argument('--terrain-types', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results/figures')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.terrain_types:
        save_path = os.path.join(args.output_dir, 'terrain_types.png')
        visualize_terrain_types(save_path)
    elif args.model:
        run_visualization(
            model_path=args.model,
            config_path=args.config,
            n_episodes=args.episodes,
            save_animation=args.animation,
            output_dir=args.output_dir
        )
    else:
        print("Please specify --model or --terrain-types")
        parser.print_help()


if __name__ == "__main__":
    main()
```

---

## 7. 使用指南

### 7.1 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练SAC模型
python train/train_sac.py --config config/config.yaml

# 3. 训练PPO模型
python train/train_ppo.py --config config/config.yaml

# 4. 评估模型
python train/evaluate.py --model results/models/sac_best.zip --episodes 100

# 5. 可视化结果
python visualization/visualizer.py --model results/models/sac_best.zip --episodes 3

# 6. 查看不同地形类型
python visualization/visualizer.py --terrain-types
```

### 7.2 TensorBoard监控

```bash
tensorboard --logdir results/logs
```

### 7.3 算法对比

| 指标 | SAC | PPO |
|------|-----|-----|
| 样本效率 | ⭐⭐⭐ | ⭐⭐ |
| 训练稳定性 | ⭐⭐ | ⭐⭐⭐ |
| 最终性能 | ⭐⭐⭐ | ⭐⭐ |
| 探索能力 | ⭐⭐⭐ | ⭐⭐ |
| 并行化 | ⭐ | ⭐⭐⭐ |

### 7.4 实验建议

1. **简单任务**：使用PPO，训练更稳定
2. **复杂地形**：使用SAC，探索能力更强
3. **有限计算资源**：使用SAC，样本效率高
4. **大规模并行**：使用PPO，支持多环境

---

## 许可证

MIT License

---

*文档生成时间: 2024年*
