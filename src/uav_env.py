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


if __name__ == "__main__":
    # 测试环境
    env = UAVPathPlanningEnv(render_mode='human')
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    total_reward = 0
    for step in range(200):
        # 随机动作
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Final info: {info}")
            break
    
    env.close()
