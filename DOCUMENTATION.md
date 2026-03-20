# DEM环境无人机路径规划强化学习实验

## 完整技术文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [环境安装](#3-环境安装)
4. [核心模块详解](#4-核心模块详解)
5. [训练指南](#5-训练指南)
6. [评估与可视化](#6-评估与可视化)
7. [算法对比分析](#7-算法对比分析)
8. [实验结果](#8-实验结果)
9. [常见问题](#9-常见问题)

---

## 1. 项目概述

### 1.1 项目目标

本项目实现了一个基于深度强化学习的无人机三维路径规划系统，主要特点：

- **真实DEM地形环境**：支持加载GeoTIFF格式DEM数据或程序化生成地形
- **物理仿真**：包含无人机动力学模型和碰撞检测
- **算法对比**：实现SAC和PPO两种主流强化学习算法
- **完整评估体系**：提供多维度性能评估和3D可视化

### 1.2 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 强化学习库 | Stable-Baselines3 |
| 环境接口 | Gymnasium |
| DEM处理 | Rasterio |
| 可视化 | Matplotlib |

### 1.3 项目结构

```
uav_path_planning/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── __init__.py          # 包初始化
│   ├── dem_loader.py        # DEM数据加载器
│   ├── uav_env.py           # Gymnasium环境
│   ├── uav_model.py         # 无人机动力学模型
│   └── utils.py             # 工具函数
├── train/
│   ├── train_sac.py         # SAC训练脚本
│   ├── train_ppo.py         # PPO训练脚本
│   └── evaluate.py          # 评估脚本
├── visualization/
│   └── visualizer.py        # 3D可视化
├── data/
│   └── dem/                 # DEM数据目录
└── results/
    ├── models/              # 训练模型
    ├── logs/                # 训练日志
    └── figures/             # 可视化图片
```

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                       系统架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                    ┌─────────────────────┐ │
│  │   DEM数据层     │                    │     可视化层        │ │
│  │  ┌───────────┐  │                    │  ┌───────────────┐  │ │
│  │  │ GeoTIFF   │  │                    │  │  3D渲染器     │  │ │
│  │  │ 地形生成器 │  │                    │  │  轨迹动画     │  │ │
│  │  └───────────┘  │                    │  │  性能图表     │  │ │
│  └────────┬────────┘                    │  └───────────────┘  │ │
│           │                             └──────────▲──────────┘ │
│           ▼                                        │            │
│  ┌─────────────────────────────────────────────────┤            │
│  │                 仿真环境层                       │            │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │            │
│  │  │  地形管理   │  │  无人机模型  │  │  碰撞检测 │ │            │
│  │  │  高度查询   │  │  状态更新   │  │  边界检测 │ │            │
│  │  └─────────────┘  └─────────────┘  └──────────┘ │            │
│  └────────────────────────┬────────────────────────┘            │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Gymnasium环境接口                          ││
│  │    ┌───────────┐   ┌────────────┐   ┌─────────────────┐     ││
│  │    │ 状态空间   │   │ 动作空间   │   │   奖励函数       │     ││
│  │    │ (14维)    │   │ (3维连续)  │   │ (多目标加权)     │     ││
│  │    └───────────┘   └────────────┘   └─────────────────┘     ││
│  └────────────────────────┬────────────────────────────────────┘│
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    强化学习算法层                            ││
│  │    ┌─────────────────────┐  ┌─────────────────────────┐     ││
│  │    │        SAC          │  │          PPO            │     ││
│  │    │  ┌───────────────┐  │  │  ┌───────────────────┐  │     ││
│  │    │  │ Actor网络     │  │  │  │ 策略网络          │  │     ││
│  │    │  │ 双Q网络       │  │  │  │ 价值网络          │  │     ││
│  │    │  │ 经验回放池    │  │  │  │ GAE优势估计       │  │     ││
│  │    │  │ 自动熵调节    │  │  │  │ 裁剪目标函数      │  │     ││
│  │    │  └───────────────┘  │  │  └───────────────────┘  │     ││
│  │    └─────────────────────┘  └─────────────────────────┘     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  观测    │────▶│  策略    │────▶│  动作    │────▶│  环境    │
│ (14维)   │     │  网络    │     │ (3维)    │     │  更新    │
└──────────┘     └──────────┘     └──────────┘     └────┬─────┘
     ▲                                                   │
     │           ┌──────────┐     ┌──────────┐          │
     │           │  奖励    │◀────│  状态    │◀─────────┘
     │           │  计算    │     │  转移    │
     │           └────┬─────┘     └──────────┘
     │                │
     └────────────────┴──────────────────────────────────
```

---

## 3. 环境安装

### 3.1 系统要求

- Python >= 3.8
- CUDA >= 11.0 (推荐，用于GPU加速)
- 8GB+ RAM

### 3.2 安装步骤

```bash
# 1. 克隆或创建项目目录
mkdir uav_path_planning && cd uav_path_planning

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; import stable_baselines3; print('OK')"
```

### 3.3 依赖说明

```
torch>=1.12.0           # PyTorch深度学习框架
stable-baselines3>=2.0  # 强化学习算法库
gymnasium>=0.29.0       # RL环境接口
numpy>=1.21.0           # 数值计算
rasterio>=1.3.0         # DEM数据读取
matplotlib>=3.5.0       # 可视化
pyyaml>=6.0             # 配置文件解析
tensorboard>=2.10.0     # 训练监控
tqdm>=4.64.0            # 进度条
```

---

## 4. 核心模块详解

### 4.1 DEM加载器 (dem_loader.py)

#### DEMLoader类

用于加载和管理DEM地形数据。

```python
from src.dem_loader import DEMLoader

# 从文件加载
loader = DEMLoader(
    dem_file="data/dem/terrain.tif",
    size_x=1000.0,      # 地形X尺寸 (米)
    size_y=1000.0,      # 地形Y尺寸 (米)
    resolution=10.0,    # 网格分辨率 (米)
    max_height=500.0    # 最大高程 (米)
)

# 查询高度
height = loader.get_elevation(x=500.0, y=300.0)
print(f"高度: {height} 米")

# 批量查询
positions = np.array([[100, 200], [300, 400], [500, 600]])
heights = loader.get_elevation_batch(positions)
```

#### TerrainGenerator类

程序化生成测试地形。

```python
from src.dem_loader import TerrainGenerator

generator = TerrainGenerator(
    size_x=1000.0,
    size_y=1000.0,
    resolution=10.0,
    max_height=500.0,
    seed=42
)

# 支持的地形类型
# - "flat": 平坦地形
# - "hills": 丘陵地形
# - "mountains": 山区地形 (Diamond-Square算法)
# - "valley": 峡谷地形
# - "mixed": 混合地形

loader = generator.generate("hills")
```

### 4.2 无人机模型 (uav_model.py)

简化的无人机动力学模型，采用速度控制。

```python
from src.uav_model import UAVModel

uav = UAVModel(
    max_velocity_xy=15.0,   # 最大水平速度 (m/s)
    max_velocity_z=5.0,     # 最大垂直速度 (m/s)
    max_acceleration=5.0,   # 最大加速度 (m/s²)
    time_constant=0.3,      # 响应时间常数 (s)
    collision_radius=2.0,   # 碰撞半径 (m)
    dt=0.1                  # 仿真步长 (s)
)

# 重置位置
uav.reset(position=np.array([100.0, 100.0, 150.0]))

# 执行动作 (速度指令，范围[-1, 1])
action = np.array([0.5, 0.0, 0.1])  # [vx, vy, vz]
state = uav.step(action)

# 获取状态
position = uav.get_position()
velocity = uav.get_velocity()

# 碰撞检测
terrain_height = 100.0
is_collision = uav.check_collision(terrain_height)
```

### 4.3 Gymnasium环境 (uav_env.py)

标准强化学习环境接口。

#### 状态空间 (14维)

| 索引 | 描述 | 范围 |
|------|------|------|
| 0-2 | 归一化位置 (x, y, z) | [0, 1] |
| 3-5 | 归一化速度 (vx, vy, vz) | [-1, 1] |
| 6-8 | 归一化相对目标位置 | [-1, 1] |
| 9 | 归一化目标距离 | [0, 1] |
| 10-13 | 归一化周围地形高度 | [0, 1] |

#### 动作空间 (3维连续)

| 索引 | 描述 | 原始范围 | 映射范围 |
|------|------|----------|----------|
| 0 | 前向速度指令 | [-1, 1] | [-15, 15] m/s |
| 1 | 侧向速度指令 | [-1, 1] | [-15, 15] m/s |
| 2 | 垂直速度指令 | [-1, 1] | [-5, 5] m/s |

#### 奖励函数

```python
def compute_reward(self):
    reward = 0.0
    
    # 1. 距离奖励 (稠密奖励，鼓励接近目标)
    distance_reward = (prev_distance - current_distance) * 10.0
    reward += distance_reward
    
    # 2. 到达目标奖励
    if reached_goal:
        reward += 500.0
    
    # 3. 碰撞惩罚
    if collision:
        reward += -200.0
    
    # 4. 高度安全奖励
    if height_above_ground < safe_altitude:
        reward += (safe_altitude - height_above_ground) * (-2.0)
    elif height_above_ground > max_altitude:
        reward += (height_above_ground - max_altitude) * (-0.5)
    
    # 5. 能耗惩罚
    reward += np.linalg.norm(action) * (-0.01)
    
    # 6. 时间惩罚
    reward += -0.1
    
    return reward
```

#### 使用示例

```python
from src.uav_env import UAVPathPlanningEnv

# 创建环境
env = UAVPathPlanningEnv(render_mode='human')

# 重置环境
obs, info = env.reset(seed=42)
print(f"观测形状: {obs.shape}")
print(f"起点: {info['position']}")
print(f"终点: {info['goal_position']}")

# 运行episode
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    done = terminated or truncated

print(f"总奖励: {total_reward}")
env.close()
```

---

## 5. 训练指南

### 5.1 配置文件说明

配置文件 `config/config.yaml` 包含所有训练参数：

```yaml
# 环境配置
environment:
  terrain:
    size_x: 1000.0          # 地形X尺寸
    size_y: 1000.0          # 地形Y尺寸
    terrain_type: "hills"   # 地形类型
  
  uav:
    max_velocity_xy: 15.0   # 最大水平速度
    max_velocity_z: 5.0     # 最大垂直速度
  
  task:
    safe_altitude: 20.0     # 安全高度
    goal_threshold: 10.0    # 到达阈值
    max_steps: 500          # 最大步数

# 奖励配置
reward:
  distance_weight: 10.0     # 距离奖励权重
  goal_reward: 500.0        # 到达奖励
  collision_penalty: -200.0 # 碰撞惩罚

# 训练配置
training:
  seed: 42
  total_timesteps: 1000000
  
  sac:
    learning_rate: 0.0003
    buffer_size: 1000000
    batch_size: 256
  
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    n_envs: 16
```

### 5.2 训练SAC

```bash
# 基本训练
python train/train_sac.py --config config/config.yaml

# 恢复训练
python train/train_sac.py --config config/config.yaml \
    --resume results/models/sac_checkpoint_500000_steps.zip
```

**SAC训练特点：**
- Off-policy算法，样本效率高
- 使用经验回放池
- 自动调节熵系数
- 适合连续动作空间

### 5.3 训练PPO

```bash
# 基本训练
python train/train_ppo.py --config config/config.yaml

# 恢复训练
python train/train_ppo.py --config config/config.yaml \
    --resume results/models/ppo_checkpoint_500000_steps.zip
```

**PPO训练特点：**
- On-policy算法
- 支持多环境并行
- 训练更稳定
- 实现简单

### 5.4 使用TensorBoard监控

```bash
tensorboard --logdir results/logs
```

在浏览器打开 `http://localhost:6006` 查看：
- Episode奖励曲线
- Episode长度曲线
- 损失函数变化
- 策略熵变化

---

## 6. 评估与可视化

### 6.1 模型评估

```bash
# 基本评估
python train/evaluate.py \
    --model results/models/sac_best.zip \
    --episodes 100

# 带渲染的评估
python train/evaluate.py \
    --model results/models/sac_best.zip \
    --episodes 10 \
    --render
```

**评估指标：**

| 指标 | 说明 |
|------|------|
| 成功率 | 到达目标的episode比例 |
| 碰撞率 | 发生碰撞的episode比例 |
| 平均奖励 | 每个episode的平均累积奖励 |
| 平均步数 | 每个episode的平均长度 |
| 路径效率 | 实际路径/直线距离 |
| 最终距离 | 结束时到目标的距离 |

### 6.2 3D可视化

```bash
# 可视化单个episode
python visualization/visualizer.py \
    --model results/models/sac_best.zip \
    --episodes 3

# 生成动画
python visualization/visualizer.py \
    --model results/models/sac_best.zip \
    --episodes 1 \
    --animation

# 可视化地形类型
python visualization/visualizer.py --terrain-types
```

### 6.3 算法对比

```python
from visualization.visualizer import UAVVisualizer

# 加载两个模型的轨迹
trajectories = [sac_trajectory, ppo_trajectory]
labels = ['SAC', 'PPO']

visualizer = UAVVisualizer(env)
visualizer.visualize_comparison(
    trajectories=trajectories,
    labels=labels,
    goal=goal_position,
    title="SAC vs PPO Comparison",
    save_path="comparison.png"
)
```

---

## 7. 算法对比分析

### 7.1 理论对比

| 方面 | SAC | PPO |
|------|-----|-----|
| 类型 | Off-policy | On-policy |
| 数据效率 | 高 | 低 |
| 探索机制 | 最大熵 | 动作噪声 |
| 训练稳定性 | 良好 | 优秀 |
| 实现复杂度 | 中等 | 简单 |
| 并行化 | 困难 | 容易 |
| 内存占用 | 高(经验池) | 低 |

### 7.2 三维路径规划场景分析

**SAC优势场景：**
- 仿真计算开销大
- 需要发现多条可行路径
- 奖励稀疏
- 环境复杂

**PPO优势场景：**
- 可大规模并行仿真
- 需要快速迭代验证
- 环境相对简单
- 稳定性要求高

### 7.3 预期性能

基于类似任务的经验，预期性能：

| 指标 | SAC | PPO |
|------|-----|-----|
| 收敛速度 | 快 | 中等 |
| 最终成功率 | 85-95% | 80-90% |
| 路径效率 | 0.7-0.85 | 0.65-0.80 |
| 训练稳定性 | 良好 | 优秀 |

---

## 8. 实验结果

### 8.1 实验设置

| 参数 | 值 |
|------|---|
| 地形尺寸 | 1000m × 1000m |
| 地形类型 | Hills |
| 训练步数 | 1,000,000 |
| 评估Episode | 100 |
| 随机种子 | 42 |

### 8.2 训练曲线

训练完成后，可在TensorBoard或保存的日志中查看：
- Episode奖励曲线
- 成功率变化
- 策略熵变化

### 8.3 结果分析模板

```
==============================================================
                    评估结果
==============================================================

📊 整体性能:
   成功率:    XX.XX%
   碰撞率:    XX.XX%
   超时率:    XX.XX%

💰 奖励统计:
   平均:  XXX.XX ± XX.XX
   范围: [XXX.XX, XXX.XX]

⏱️ Episode长度:
   平均:  XXX.X ± XX.X

🎯 成功Episode:
   平均奖励:      XXX.XX
   平均长度:      XXX.X
   平均路径长度: XXX.X m
   平均效率:      X.XX

📍 最终距离:
   平均: XX.XX ± XX.XX m
==============================================================
```

---

## 9. 常见问题

### Q1: 训练不收敛怎么办？

**可能原因及解决方案：**

1. **奖励函数设计不当**
   - 增加距离奖励权重
   - 减少惩罚项权重
   - 添加更多形成性奖励

2. **学习率过大/过小**
   - SAC建议: 1e-4 到 3e-4
   - PPO建议: 1e-4 到 3e-4

3. **网络容量不足**
   - 增加网络宽度: [256, 256] → [512, 512]
   - 增加网络深度: 2层 → 3层

4. **探索不足**
   - SAC: 检查熵系数是否正常
   - PPO: 增加ent_coef

### Q2: 内存不足怎么办？

**解决方案：**
1. 减少SAC的buffer_size
2. 减少PPO的n_envs
3. 减少batch_size
4. 使用更小的网络

### Q3: 如何使用自定义DEM数据？

```python
# 1. 准备GeoTIFF格式的DEM文件
# 2. 修改配置文件
environment:
  terrain:
    dem_file: "data/dem/your_terrain.tif"
    size_x: 1000.0  # 根据实际调整
    size_y: 1000.0
```

### Q4: 如何调整难度？

**降低难度：**
- 使用"flat"地形
- 增加goal_threshold
- 减少max_steps
- 增加safe_altitude

**提高难度：**
- 使用"mountains"地形
- 减少goal_threshold
- 增加max_steps
- 减少safe_altitude
- 增加起终点距离

---

## 附录

### A. 完整配置参数表

见 `config/config.yaml`

### B. API参考

见各模块的docstring

### C. 更新日志

- v1.0.0: 初始版本，实现SAC和PPO训练

---

*文档最后更新: 2024年*
