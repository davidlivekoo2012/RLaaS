"""
Policy Engine for reinforcement learning agents.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from .agents import SACAgent, PPOAgent
from .environments import NetworkEnvironment, RecommendationEnvironment
from rlaas.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class PolicyType(Enum):
    """Types of RL policies."""
    SAC = "sac"
    PPO = "ppo"


class EnvironmentType(Enum):
    """Types of environments."""
    NETWORK_5G = "network_5g"
    RECOMMENDATION = "recommendation"


@dataclass
class PolicyConfig:
    """Configuration for policy training."""
    policy_type: PolicyType
    environment_type: EnvironmentType
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    policy_kwargs: Optional[Dict[str, Any]] = None
    device: str = "auto"


@dataclass
class PolicyResult:
    """Result of policy training or inference."""
    policy_id: str
    rewards: List[float]
    episode_lengths: List[int]
    training_time: float
    final_reward: float
    convergence_achieved: bool
    metadata: Dict[str, Any]


class PolicyEngine:
    """
    Main policy engine for reinforcement learning.
    
    Manages RL agents, environments, and training processes for
    5G network optimization and recommendation systems.
    """
    
    def __init__(self):
        self.agents: Dict[str, Union[SACAgent, PPOAgent]] = {}
        self.environments: Dict[str, Any] = {}
        self._training_tasks: Dict[str, asyncio.Task] = {}
    
    async def create_policy(
        self,
        policy_id: str,
        config: PolicyConfig
    ) -> str:
        """
        Create a new RL policy.
        
        Args:
            policy_id: Unique identifier for the policy
            config: Policy configuration
            
        Returns:
            Policy ID
        """
        logger.info(f"Creating policy {policy_id} with type {config.policy_type.value}")
        
        # Create environment
        env = self._create_environment(config.environment_type)
        self.environments[policy_id] = env
        
        # Create agent
        if config.policy_type == PolicyType.SAC:
            agent = SACAgent(
                env=env,
                learning_rate=config.learning_rate,
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
                gamma=config.gamma,
                tau=config.tau,
                policy_kwargs=config.policy_kwargs,
                device=config.device
            )
        elif config.policy_type == PolicyType.PPO:
            agent = PPOAgent(
                env=env,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                gamma=config.gamma,
                policy_kwargs=config.policy_kwargs,
                device=config.device
            )
        else:
            raise ValueError(f"Unsupported policy type: {config.policy_type}")
        
        self.agents[policy_id] = agent
        
        logger.info(f"Policy {policy_id} created successfully")
        return policy_id
    
    async def train_policy(
        self,
        policy_id: str,
        config: PolicyConfig,
        background: bool = True
    ) -> Union[PolicyResult, str]:
        """
        Train a reinforcement learning policy.
        
        Args:
            policy_id: Policy identifier
            config: Training configuration
            background: Whether to run training in background
            
        Returns:
            Training result or task ID if background
        """
        if policy_id not in self.agents:
            raise ValueError(f"Policy {policy_id} not found")
        
        if background:
            # Start training in background
            task = asyncio.create_task(
                self._train_policy_async(policy_id, config)
            )
            self._training_tasks[policy_id] = task
            return f"training_{policy_id}"
        else:
            # Train synchronously
            return await self._train_policy_async(policy_id, config)
    
    async def _train_policy_async(
        self,
        policy_id: str,
        config: PolicyConfig
    ) -> PolicyResult:
        """Internal async training method."""
        
        start_time = asyncio.get_event_loop().time()
        agent = self.agents[policy_id]
        
        logger.info(f"Starting training for policy {policy_id}")
        
        # Training callback to track progress
        class TrainingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode_rewards = []
                self.episode_lengths = []
            
            def _on_step(self) -> bool:
                if len(self.locals.get('infos', [])) > 0:
                    for info in self.locals['infos']:
                        if 'episode' in info:
                            self.episode_rewards.append(info['episode']['r'])
                            self.episode_lengths.append(info['episode']['l'])
                return True
        
        callback = TrainingCallback()
        
        # Train the agent
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.learn(
                total_timesteps=config.total_timesteps,
                callback=callback
            )
        )
        
        training_time = asyncio.get_event_loop().time() - start_time
        
        # Calculate results
        final_reward = np.mean(callback.episode_rewards[-10:]) if callback.episode_rewards else 0.0
        convergence_achieved = len(callback.episode_rewards) > 50 and \
                             np.std(callback.episode_rewards[-20:]) < 0.1 * abs(final_reward)
        
        result = PolicyResult(
            policy_id=policy_id,
            rewards=callback.episode_rewards,
            episode_lengths=callback.episode_lengths,
            training_time=training_time,
            final_reward=final_reward,
            convergence_achieved=convergence_achieved,
            metadata={
                "total_timesteps": config.total_timesteps,
                "policy_type": config.policy_type.value,
                "environment_type": config.environment_type.value,
                "learning_rate": config.learning_rate,
            }
        )
        
        logger.info(f"Training completed for policy {policy_id}. Final reward: {final_reward:.4f}")
        
        return result
    
    async def predict(
        self,
        policy_id: str,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Make prediction using trained policy.
        
        Args:
            policy_id: Policy identifier
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action prediction
        """
        if policy_id not in self.agents:
            raise ValueError(f"Policy {policy_id} not found")
        
        agent = self.agents[policy_id]
        action, _ = agent.predict(observation, deterministic=deterministic)
        
        return action
    
    async def evaluate_policy(
        self,
        policy_id: str,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate a trained policy.
        
        Args:
            policy_id: Policy identifier
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        if policy_id not in self.agents:
            raise ValueError(f"Policy {policy_id} not found")
        
        agent = self.agents[policy_id]
        env = self.environments[policy_id]
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
        }
    
    async def save_policy(
        self,
        policy_id: str,
        path: str
    ) -> str:
        """
        Save a trained policy to disk.
        
        Args:
            policy_id: Policy identifier
            path: Save path
            
        Returns:
            Saved file path
        """
        if policy_id not in self.agents:
            raise ValueError(f"Policy {policy_id} not found")
        
        agent = self.agents[policy_id]
        agent.save(path)
        
        logger.info(f"Policy {policy_id} saved to {path}")
        return path
    
    async def load_policy(
        self,
        policy_id: str,
        path: str,
        environment_type: EnvironmentType
    ) -> str:
        """
        Load a trained policy from disk.
        
        Args:
            policy_id: Policy identifier
            path: Load path
            environment_type: Environment type
            
        Returns:
            Policy ID
        """
        # Create environment
        env = self._create_environment(environment_type)
        self.environments[policy_id] = env
        
        # Determine agent type from path or metadata
        # For simplicity, assume SAC if not specified
        agent = SACAgent.load(path, env=env)
        self.agents[policy_id] = agent
        
        logger.info(f"Policy {policy_id} loaded from {path}")
        return policy_id
    
    def _create_environment(self, env_type: EnvironmentType):
        """Create environment based on type."""
        
        if env_type == EnvironmentType.NETWORK_5G:
            return NetworkEnvironment()
        elif env_type == EnvironmentType.RECOMMENDATION:
            return RecommendationEnvironment()
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
    
    async def get_training_status(self, policy_id: str) -> Dict[str, Any]:
        """Get training status for a policy."""
        
        if policy_id not in self._training_tasks:
            return {"status": "not_found"}
        
        task = self._training_tasks[policy_id]
        
        if task.done():
            if task.exception():
                return {
                    "status": "failed",
                    "error": str(task.exception())
                }
            else:
                result = task.result()
                return {
                    "status": "completed",
                    "result": result
                }
        else:
            return {"status": "running"}
    
    async def cancel_training(self, policy_id: str) -> bool:
        """Cancel training for a policy."""
        
        if policy_id not in self._training_tasks:
            return False
        
        task = self._training_tasks[policy_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self._training_tasks[policy_id]
        return True
    
    def list_policies(self) -> List[str]:
        """List all available policies."""
        return list(self.agents.keys())
