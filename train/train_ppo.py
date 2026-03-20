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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

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
    """
    创建并行环境
    
    Args:
        config: 配置字典
        n_envs: 并行环境数量
        seed: 随机种子
        
    Returns:
        向量化环境
    """
    env_fns = [make_env(config, rank=i, seed=seed) for i in range(n_envs)]
    
    # 使用子进程向量化环境 (更快但内存占用更大)
    # 如果内存有限,使用 DummyVecEnv
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
    print(f"Figure directory: {dirs['figure_dir']}")
    
    # 保存配置
    save_config(config, os.path.join(dirs['log_dir'], 'config.yaml'))
    
    # 打印配置
    print(f"\n{'='*60}")
    print("Configuration:")
    print(f"{'='*60}")
    print_config(config)
    
    # 获取设备
    device = get_device(training_config.get('device', 'auto'))
    print(f"\nUsing device: {device}")
    
    # 创建并行环境
    n_envs = ppo_config.get('n_envs', 8)
    print(f"\nCreating {n_envs} parallel environments...")
    env = create_parallel_envs(config, n_envs, seed)
    
    # 可选: 环境归一化
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
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
    
    # 打印模型信息
    print(f"\nModel architecture:")
    print(model.policy)
    
    # 计算实际的评估和保存频率 (考虑并行环境)
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
    print(f"Steps per update: {ppo_config['n_steps'] * n_envs}")
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
    print(f"Final model: {final_model_path}")
    print(f"Best model: {os.path.join(dirs['model_dir'], 'best_model.zip')}")
    
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
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # 开始训练
    train_ppo(args.config, args.resume)


if __name__ == "__main__":
    main()
