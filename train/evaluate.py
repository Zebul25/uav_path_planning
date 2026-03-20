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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor

from src.uav_env import UAVPathPlanningEnv
from src.utils import (
    load_config,
    set_seed,
    calculate_path_metrics
)


def load_model(model_path: str, env):
    """
    加载模型
    
    Args:
        model_path: 模型路径
        env: 环境实例
        
    Returns:
        加载的模型
    """
    # 根据文件名推断算法类型
    if 'sac' in model_path.lower():
        model = SAC.load(model_path, env=env)
        print(f"Loaded SAC model from {model_path}")
    elif 'ppo' in model_path.lower():
        model = PPO.load(model_path, env=env)
        print(f"Loaded PPO model from {model_path}")
    else:
        # 尝试两种算法
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
    """
    评估模型
    
    Args:
        model: 训练好的模型
        env: 评估环境
        n_episodes: 评估episode数量
        deterministic: 是否使用确定性策略
        render: 是否渲染
        save_trajectories: 是否保存轨迹
        
    Returns:
        评估结果字典
    """
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
        
        # 保存轨迹
        if save_trajectories:
            results['trajectories'].append(trajectory.tolist())
    
    return results


def compute_statistics(results: dict) -> dict:
    """
    计算统计指标
    
    Args:
        results: 评估结果
        
    Returns:
        统计指标字典
    """
    stats = {}
    
    # 成功率
    stats['success_rate'] = np.mean(results['success']) * 100
    
    # 碰撞率
    stats['collision_rate'] = np.mean(results['collision']) * 100
    
    # 超时率
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


def plot_results(results: dict, save_path: str = None):
    """
    绘制评估结果
    
    Args:
        results: 评估结果
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 奖励分布
    ax = axes[0, 0]
    ax.hist(results['episode_rewards'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(results['episode_rewards']), color='r', linestyle='--', label='Mean')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution')
    ax.legend()
    
    # 2. Episode长度分布
    ax = axes[0, 1]
    ax.hist(results['episode_lengths'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(results['episode_lengths']), color='r', linestyle='--', label='Mean')
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Count')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    
    # 3. 结果类型饼图
    ax = axes[0, 2]
    labels = ['Success', 'Collision', 'Timeout']
    sizes = [
        np.sum(results['success']),
        np.sum(results['collision']),
        np.sum(results['timeout'])
    ]
    colors = ['green', 'red', 'orange']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Episode Outcomes')
    
    # 4. 最终距离分布
    ax = axes[1, 0]
    ax.hist(results['final_distances'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(results['final_distances']), color='r', linestyle='--', label='Mean')
    ax.set_xlabel('Final Distance (m)')
    ax.set_ylabel('Count')
    ax.set_title('Final Distance Distribution')
    ax.legend()
    
    # 5. 路径长度 vs Episode长度
    ax = axes[1, 1]
    ax.scatter(results['episode_lengths'], results['path_lengths'], alpha=0.5)
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Path Length (m)')
    ax.set_title('Path Length vs Episode Length')
    
    # 6. 奖励随Episode变化
    ax = axes[1, 2]
    rewards = results['episode_rewards']
    window = min(10, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(rewards, alpha=0.3, label='Raw')
    ax.plot(range(window-1, len(rewards)), smoothed, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward over Episodes')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_trajectories(env, results: dict, n_trajectories: int = 5, save_path: str = None):
    """
    绘制3D轨迹
    
    Args:
        env: 环境实例
        results: 评估结果
        n_trajectories: 绘制的轨迹数量
        save_path: 保存路径
    """
    if results['trajectories'] is None or len(results['trajectories']) == 0:
        print("No trajectories to plot")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制地形
    terrain_grid = env.terrain.get_elevation_grid()
    x = np.linspace(0, env.terrain_size_x, terrain_grid.shape[1])
    y = np.linspace(0, env.terrain_size_y, terrain_grid.shape[0])
    X, Y = np.meshgrid(x, y)
    
    ax.plot_surface(X, Y, terrain_grid, cmap='terrain', alpha=0.5)
    
    # 选择轨迹 (优先选择成功的)
    success_indices = [i for i, s in enumerate(results['success']) if s]
    if len(success_indices) >= n_trajectories:
        indices = np.random.choice(success_indices, n_trajectories, replace=False)
    else:
        indices = np.random.choice(len(results['trajectories']), 
                                   min(n_trajectories, len(results['trajectories'])), 
                                   replace=False)
    
    # 绘制轨迹
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))
    for idx, color in zip(indices, colors):
        traj = np.array(results['trajectories'][idx])
        label = 'Success' if results['success'][idx] else 'Failed'
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, linewidth=2, label=f'{label} {idx}')
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                   color='green', s=100, marker='o')
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                   color='red', s=100, marker='x')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Sample Trajectories')
    ax.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Trajectory figure saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
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
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render during evaluation'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='results/evaluation',
        help='Directory to save results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建环境
    render_mode = 'human' if args.render else None
    env = UAVPathPlanningEnv(config=config, render_mode=render_mode)
    env = Monitor(env)
    
    # 加载模型
    model = load_model(args.model, env)
    
    # 评估模型
    results = evaluate_model(
        model=model,
        env=env,
        n_episodes=args.episodes,
        deterministic=True,
        render=args.render,
        save_trajectories=True
    )
    
    # 计算统计
    stats = compute_statistics(results)
    
    # 打印统计
    print_statistics(stats)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存结果
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        # 不保存轨迹到JSON (太大)
        results_save = {k: v for k, v in results.items() if k != 'trajectories'}
        json.dump(results_save, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # 保存统计
    stats_path = os.path.join(save_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")
    
    # 绘制结果
    plot_results(results, save_path=os.path.join(save_dir, 'results.png'))
    plot_trajectories(env, results, n_trajectories=5, 
                     save_path=os.path.join(save_dir, 'trajectories.png'))
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
