"""
3D可视化工具
============

提供无人机路径规划的3D可视化,包括:
1. 地形渲染
2. 轨迹动画
3. 实时仿真可视化
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC, PPO
from src.uav_env import UAVPathPlanningEnv
from src.utils import load_config, set_seed


class UAVVisualizer:
    """
    无人机路径规划可视化器
    """
    
    def __init__(
        self,
        env: UAVPathPlanningEnv,
        figsize: tuple = (12, 8),
        colormap: str = 'terrain'
    ):
        """
        初始化可视化器
        
        Args:
            env: 环境实例
            figsize: 图形大小
            colormap: 地形颜色映射
        """
        self.env = env
        self.figsize = figsize
        self.colormap = colormap
        
        # 获取地形数据
        self.terrain_grid = env.terrain.get_elevation_grid()
        self.terrain_x = np.linspace(0, env.terrain_size_x, self.terrain_grid.shape[1])
        self.terrain_y = np.linspace(0, env.terrain_size_y, self.terrain_grid.shape[0])
        self.X, self.Y = np.meshgrid(self.terrain_x, self.terrain_y)
        
        # 图形对象
        self.fig = None
        self.ax = None
        self.trajectory_line = None
        self.uav_marker = None
        self.goal_marker = None
        
    def setup_figure(self):
        """设置图形"""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置视角
        self.ax.view_init(elev=30, azim=45)
        
        return self.fig, self.ax
    
    def render_terrain(self, alpha: float = 0.6):
        """
        渲染地形
        
        Args:
            alpha: 透明度
        """
        surf = self.ax.plot_surface(
            self.X, self.Y, self.terrain_grid,
            cmap=self.colormap,
            alpha=alpha,
            antialiased=True,
            shade=True
        )
        return surf
    
    def render_trajectory(
        self,
        trajectory: np.ndarray,
        color: str = 'blue',
        linewidth: float = 2,
        label: str = 'Trajectory'
    ):
        """
        渲染轨迹
        
        Args:
            trajectory: 轨迹数组, shape (N, 3)
            color: 颜色
            linewidth: 线宽
            label: 标签
        """
        line, = self.ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            linewidth=linewidth,
            label=label
        )
        return line
    
    def render_uav(
        self,
        position: np.ndarray,
        size: float = 100,
        color: str = 'blue'
    ):
        """
        渲染无人机位置
        
        Args:
            position: 位置 [x, y, z]
            size: 标记大小
            color: 颜色
        """
        marker = self.ax.scatter(
            position[0], position[1], position[2],
            c=color, s=size, marker='o', label='UAV'
        )
        return marker
    
    def render_goal(
        self,
        position: np.ndarray,
        size: float = 200,
        color: str = 'red'
    ):
        """
        渲染目标位置
        
        Args:
            position: 位置 [x, y, z]
            size: 标记大小
            color: 颜色
        """
        marker = self.ax.scatter(
            position[0], position[1], position[2],
            c=color, s=size, marker='*', label='Goal'
        )
        return marker
    
    def set_labels(self, title: str = "UAV Path Planning"):
        """设置标签"""
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(title)
        self.ax.legend(loc='upper right')
    
    def visualize_episode(
        self,
        trajectory: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        title: str = "UAV Path",
        save_path: str = None
    ):
        """
        可视化单个episode
        
        Args:
            trajectory: 轨迹
            start: 起点
            goal: 终点
            title: 标题
            save_path: 保存路径
        """
        self.setup_figure()
        
        # 渲染地形
        self.render_terrain()
        
        # 渲染轨迹
        self.render_trajectory(trajectory)
        
        # 渲染起点和终点
        self.ax.scatter(start[0], start[1], start[2], 
                       c='green', s=150, marker='o', label='Start')
        self.render_goal(goal)
        
        # 设置标签
        self.set_labels(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def visualize_comparison(
        self,
        trajectories: list,
        labels: list,
        goal: np.ndarray,
        title: str = "Algorithm Comparison",
        save_path: str = None
    ):
        """
        比较多条轨迹
        
        Args:
            trajectories: 轨迹列表
            labels: 标签列表
            goal: 终点
            title: 标题
            save_path: 保存路径
        """
        self.setup_figure()
        
        # 渲染地形
        self.render_terrain(alpha=0.4)
        
        # 颜色映射
        colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
        
        # 渲染轨迹
        for traj, label, color in zip(trajectories, labels, colors):
            self.render_trajectory(traj, color=color, label=label)
            # 起点
            self.ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                           c=[color], s=100, marker='o')
        
        # 渲染终点
        self.render_goal(goal)
        
        # 设置标签
        self.set_labels(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def create_animation(
        self,
        trajectory: np.ndarray,
        goal: np.ndarray,
        fps: int = 30,
        interval: int = 50,
        save_path: str = None
    ):
        """
        创建轨迹动画
        
        Args:
            trajectory: 轨迹
            goal: 终点
            fps: 帧率
            interval: 帧间隔(ms)
            save_path: 保存路径
        """
        self.setup_figure()
        
        # 渲染静态元素
        self.render_terrain(alpha=0.5)
        self.render_goal(goal)
        
        # 初始化动态元素
        line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        uav, = self.ax.plot([], [], [], 'bo', markersize=10)
        
        # 设置标签
        self.set_labels("UAV Flight Animation")
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            uav.set_data([], [])
            uav.set_3d_properties([])
            return line, uav
        
        def update(frame):
            # 更新轨迹
            line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
            line.set_3d_properties(trajectory[:frame+1, 2])
            
            # 更新无人机位置
            uav.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
            uav.set_3d_properties([trajectory[frame, 2]])
            
            return line, uav
        
        anim = FuncAnimation(
            self.fig, update,
            frames=len(trajectory),
            init_func=init,
            interval=interval,
            blit=True
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


def visualize_terrain_types(save_path: str = None):
    """
    可视化不同地形类型
    
    Args:
        save_path: 保存路径
    """
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


def run_visualization(
    model_path: str,
    config_path: str,
    n_episodes: int = 1,
    save_animation: bool = False,
    output_dir: str = 'results/figures'
):
    """
    运行可视化
    
    Args:
        model_path: 模型路径
        config_path: 配置文件路径
        n_episodes: 可视化的episode数量
        save_animation: 是否保存动画
        output_dir: 输出目录
    """
    # 加载配置
    config = load_config(config_path)
    set_seed(config['training'].get('seed', 42))
    
    # 创建环境
    env = UAVPathPlanningEnv(config=config)
    
    # 加载模型
    if 'sac' in model_path.lower():
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)
    
    # 创建可视化器
    visualizer = UAVVisualizer(env)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for ep in range(n_episodes):
        print(f"\nEpisode {ep + 1}/{n_episodes}")
        
        # 运行episode
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
        
        # 可视化
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
        
        # 创建动画
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
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to visualize'
    )
    parser.add_argument(
        '--animation',
        action='store_true',
        help='Save animation'
    )
    parser.add_argument(
        '--terrain-types',
        action='store_true',
        help='Visualize different terrain types'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/figures',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.terrain_types:
        # 可视化地形类型
        save_path = os.path.join(args.output_dir, 'terrain_types.png')
        visualize_terrain_types(save_path)
    elif args.model:
        # 运行模型可视化
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
