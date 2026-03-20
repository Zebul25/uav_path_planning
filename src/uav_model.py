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


class UAVModelAdvanced(UAVModel):
    """
    高级无人机模型
    
    扩展基础模型,增加:
    - 姿态控制
    - 风扰动
    - 更精确的动力学
    """
    
    def __init__(
        self,
        max_velocity_xy: float = 15.0,
        max_velocity_z: float = 5.0,
        max_acceleration: float = 5.0,
        max_roll: float = 30.0,  # 最大滚转角 (度)
        max_pitch: float = 30.0,  # 最大俯仰角 (度)
        max_yaw_rate: float = 90.0,  # 最大偏航角速度 (度/s)
        time_constant: float = 0.3,
        collision_radius: float = 2.0,
        dt: float = 0.1,
        wind_enabled: bool = False
    ):
        super().__init__(
            max_velocity_xy=max_velocity_xy,
            max_velocity_z=max_velocity_z,
            max_acceleration=max_acceleration,
            time_constant=time_constant,
            collision_radius=collision_radius,
            dt=dt
        )
        
        self.max_roll = np.radians(max_roll)
        self.max_pitch = np.radians(max_pitch)
        self.max_yaw_rate = np.radians(max_yaw_rate)
        self.wind_enabled = wind_enabled
        
        # 姿态状态 [roll, pitch, yaw]
        self.attitude = np.zeros(3)
        
        # 风速
        self.wind_velocity = np.zeros(3)
        
    def reset(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        attitude: Optional[np.ndarray] = None
    ) -> UAVState:
        """重置状态"""
        super().reset(position, velocity)
        self.attitude = attitude if attitude is not None else np.zeros(3)
        return self.state.copy()
    
    def set_wind(self, wind_velocity: np.ndarray):
        """设置风速"""
        self.wind_velocity = wind_velocity.copy()
    
    def step(self, action: np.ndarray) -> UAVState:
        """
        执行一步仿真
        
        Args:
            action: 速度指令 [vx_cmd, vy_cmd, vz_cmd], 范围 [-1, 1]
            
        Returns:
            更新后的状态
        """
        # 基础动力学更新
        state = super().step(action)
        
        # 添加风扰动
        if self.wind_enabled:
            wind_effect = self.wind_velocity * self.dt
            self.state.position += wind_effect
        
        # 更新姿态 (简化模型: 根据速度方向)
        speed_xy = np.linalg.norm(self.state.velocity[:2])
        if speed_xy > 0.1:
            self.attitude[2] = np.arctan2(self.state.velocity[1], self.state.velocity[0])
        
        # 根据水平速度估算俯仰角
        speed = np.linalg.norm(self.state.velocity)
        if speed > 0.1:
            self.attitude[1] = np.clip(
                np.arctan2(self.state.velocity[2], speed_xy),
                -self.max_pitch,
                self.max_pitch
            )
        
        return self.state.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """获取状态信息字典"""
        info = super().get_info()
        info['attitude'] = np.degrees(self.attitude).tolist()
        info['wind_velocity'] = self.wind_velocity.tolist()
        return info


if __name__ == "__main__":
    # 测试无人机模型
    import matplotlib.pyplot as plt
    
    # 创建模型
    uav = UAVModel(dt=0.1)
    uav.reset(position=np.array([0.0, 0.0, 100.0]))
    
    # 仿真轨迹
    positions = []
    velocities = []
    
    # 直线飞行
    for _ in range(50):
        uav.step(np.array([1.0, 0.0, 0.0]))
        positions.append(uav.get_position().copy())
        velocities.append(uav.get_velocity().copy())
    
    # 转弯
    for _ in range(30):
        uav.step(np.array([0.5, 0.5, 0.1]))
        positions.append(uav.get_position().copy())
        velocities.append(uav.get_velocity().copy())
    
    # 下降
    for _ in range(20):
        uav.step(np.array([0.3, 0.0, -0.5]))
        positions.append(uav.get_position().copy())
        velocities.append(uav.get_velocity().copy())
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # 绘图
    fig = plt.figure(figsize=(15, 5))
    
    # 3D轨迹
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # 位置
    ax2 = fig.add_subplot(132)
    time = np.arange(len(positions)) * 0.1
    ax2.plot(time, positions[:, 0], label='X')
    ax2.plot(time, positions[:, 1], label='Y')
    ax2.plot(time, positions[:, 2], label='Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # 速度
    ax3 = fig.add_subplot(133)
    ax3.plot(time, velocities[:, 0], label='Vx')
    ax3.plot(time, velocities[:, 1], label='Vy')
    ax3.plot(time, velocities[:, 2], label='Vz')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig("uav_model_test.png", dpi=150)
    plt.show()
