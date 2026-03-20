# DEM环境无人机路径规划强化学习实验

## 项目概述

本项目实现了基于深度强化学习的无人机三维路径规划系统，在真实DEM（数字高程模型）地形环境中训练无人机自主规划安全路径。项目对比了SAC和PPO两种主流强化学习算法的性能表现。

## 项目结构

```
uav_path_planning/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖包
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── __init__.py
│   ├── dem_loader.py        # DEM数据加载器
│   ├── uav_env.py           # Gym环境实现
│   ├── uav_model.py         # 无人机动力学模型
│   └── utils.py             # 工具函数
├── train/
│   ├── train_sac.py         # SAC训练脚本
│   ├── train_ppo.py         # PPO训练脚本
│   └── evaluate.py          # 评估脚本
├── visualization/
│   └── visualizer.py        # 3D可视化
├── data/
│   └── dem/                 # DEM数据存放目录
└── results/
    ├── models/              # 训练好的模型
    ├── logs/                # 训练日志
    └── figures/             # 可视化图片
```

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖

- Python >= 3.8
- PyTorch >= 1.12
- Stable-Baselines3 >= 2.0
- Gymnasium >= 0.29
- NumPy, Matplotlib
- Rasterio (DEM读取)
- PyYAML

## 快速开始

### 1. 准备DEM数据

将GeoTIFF格式的DEM文件放入 `data/dem/` 目录，或使用生成的模拟地形。

### 2. 训练模型

```bash
# 训练SAC
python train/train_sac.py --config config/config.yaml

# 训练PPO
python train/train_ppo.py --config config/config.yaml
```

### 3. 评估与可视化

```bash
# 评估模型
python train/evaluate.py --model results/models/sac_best.zip

# 可视化路径
python visualization/visualizer.py --model results/models/sac_best.zip
```

## 实验设计

### 状态空间 (14维)

| 维度 | 描述 | 范围 |
|------|------|------|
| 0-2 | 无人机位置 (x, y, z) | 归一化到[0,1] |
| 3-5 | 无人机速度 (vx, vy, vz) | 归一化到[-1,1] |
| 6-8 | 相对目标位置 (dx, dy, dz) | 归一化 |
| 9 | 到目标距离 | 归一化 |
| 10-13 | 周围地形高度 (前/左/右/下) | 归一化 |

### 动作空间 (3维连续)

| 维度 | 描述 | 范围 |
|------|------|------|
| 0 | 前向速度指令 | [-1, 1] → [-15, 15] m/s |
| 1 | 侧向速度指令 | [-1, 1] → [-15, 15] m/s |
| 2 | 垂直速度指令 | [-1, 1] → [-5, 5] m/s |

### 奖励函数

- **距离奖励**: 接近目标获得正奖励
- **到达奖励**: +500
- **碰撞惩罚**: -200
- **高度安全奖励**: 保持安全高度
- **能耗惩罚**: 动作幅度惩罚
- **时间惩罚**: 每步-0.1

## 算法对比

| 指标 | SAC | PPO |
|------|-----|-----|
| 样本效率 | ⭐⭐⭐ | ⭐⭐ |
| 训练稳定性 | ⭐⭐ | ⭐⭐⭐ |
| 最终性能 | ⭐⭐⭐ | ⭐⭐ |
| 探索能力 | ⭐⭐⭐ | ⭐⭐ |

## 许可证

MIT License
