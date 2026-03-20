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


class TensorboardCallback:
    """自定义Tensorboard回调"""
    
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self):
        return True


def create_env(config, seed=0):
    """
    创建训练环境
    
    Args:
        config: 配置字典
        seed: 随机种子
        
    Returns:
        环境实例
    """
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
    
    # 创建环境
    print("\nCreating environment...")
    env = DummyVecEnv([lambda: create_env(config, seed)])
    
    # 可选: 环境归一化
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
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
    
    # 打印模型信息
    print(f"\nModel architecture:")
    print(model.policy)
    
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
    print(f"Final model: {final_model_path}")
    print(f"Best model: {os.path.join(dirs['model_dir'], 'best_model.zip')}")
    
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
