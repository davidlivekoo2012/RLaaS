"""
Reinforcement Learning agents for RLaaS platform.
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

logger = logging.getLogger(__name__)


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for network and recommendation environments.
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimension
        input_dim = observation_space.shape[0]
        
        # Neural network layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_net(observations)


class SACAgent:
    """
    Soft Actor-Critic agent for continuous action spaces.
    
    Suitable for 5G network optimization with continuous parameters
    like power allocation and beamforming weights.
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: str = "auto",
        target_update_interval: int = 1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "auto"
    ):
        """
        Initialize SAC agent.
        
        Args:
            env: Environment instance
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Soft update coefficient
            ent_coef: Entropy coefficient
            target_update_interval: Target network update interval
            policy_kwargs: Policy network arguments
            device: Device to use (cpu/cuda)
        """
        
        # Default policy kwargs with custom feature extractor
        if policy_kwargs is None:
            policy_kwargs = {
                "features_extractor_class": CustomFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
                "net_arch": [256, 256],
                "activation_fn": torch.nn.ReLU,
            }
        
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1
        )
        
        self.env = env
        logger.info("SAC agent initialized")
    
    def learn(self, total_timesteps: int, callback=None) -> "SACAgent":
        """Train the agent."""
        logger.info(f"Starting SAC training for {total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=1000
        )
        return self
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make prediction."""
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
        logger.info(f"SAC model saved to {path}")
    
    @classmethod
    def load(cls, path: str, env=None) -> "SACAgent":
        """Load a saved model."""
        model = SAC.load(path, env=env)
        agent = cls.__new__(cls)
        agent.model = model
        agent.env = env
        logger.info(f"SAC model loaded from {path}")
        return agent


class PPOAgent:
    """
    Proximal Policy Optimization agent for discrete action spaces.
    
    Suitable for recommendation systems with discrete actions
    like item selection and ranking strategies.
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "auto"
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: Environment instance
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: Clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            policy_kwargs: Policy network arguments
            device: Device to use (cpu/cuda)
        """
        
        # Default policy kwargs with custom feature extractor
        if policy_kwargs is None:
            policy_kwargs = {
                "features_extractor_class": CustomFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
                "activation_fn": torch.nn.ReLU,
            }
        
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1
        )
        
        self.env = env
        logger.info("PPO agent initialized")
    
    def learn(self, total_timesteps: int, callback=None) -> "PPOAgent":
        """Train the agent."""
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=1000
        )
        return self
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make prediction."""
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
        logger.info(f"PPO model saved to {path}")
    
    @classmethod
    def load(cls, path: str, env=None) -> "PPOAgent":
        """Load a saved model."""
        model = PPO.load(path, env=env)
        agent = cls.__new__(cls)
        agent.model = model
        agent.env = env
        logger.info(f"PPO model loaded from {path}")
        return agent


class MultiAgentSystem:
    """
    Multi-agent system for coordinated optimization.
    
    Manages multiple RL agents working together on complex
    optimization problems.
    """
    
    def __init__(self):
        self.agents: Dict[str, Union[SACAgent, PPOAgent]] = {}
        self.coordination_strategy = "independent"  # independent, cooperative, competitive
    
    def add_agent(self, agent_id: str, agent: Union[SACAgent, PPOAgent]):
        """Add an agent to the system."""
        self.agents[agent_id] = agent
        logger.info(f"Agent {agent_id} added to multi-agent system")
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the system."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} removed from multi-agent system")
    
    def coordinate_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Coordinate actions across multiple agents.
        
        Args:
            observations: Observations for each agent
            deterministic: Whether to use deterministic policies
            
        Returns:
            Actions for each agent
        """
        actions = {}
        
        if self.coordination_strategy == "independent":
            # Independent action selection
            for agent_id, obs in observations.items():
                if agent_id in self.agents:
                    action, _ = self.agents[agent_id].predict(obs, deterministic)
                    actions[agent_id] = action
        
        elif self.coordination_strategy == "cooperative":
            # Cooperative action selection with communication
            # This is a simplified version - in practice, you'd implement
            # more sophisticated coordination mechanisms
            for agent_id, obs in observations.items():
                if agent_id in self.agents:
                    # Add information from other agents to observation
                    enhanced_obs = self._enhance_observation_with_coordination(
                        obs, agent_id, observations
                    )
                    action, _ = self.agents[agent_id].predict(enhanced_obs, deterministic)
                    actions[agent_id] = action
        
        return actions
    
    def _enhance_observation_with_coordination(
        self,
        obs: np.ndarray,
        agent_id: str,
        all_observations: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Enhance observation with coordination information.
        
        This is a placeholder for more sophisticated coordination mechanisms.
        """
        # Simple approach: concatenate mean of other agents' observations
        other_obs = [
            other_obs for other_id, other_obs in all_observations.items()
            if other_id != agent_id
        ]
        
        if other_obs:
            mean_other_obs = np.mean(other_obs, axis=0)
            enhanced_obs = np.concatenate([obs, mean_other_obs])
        else:
            enhanced_obs = obs
        
        return enhanced_obs
    
    def set_coordination_strategy(self, strategy: str):
        """Set coordination strategy."""
        if strategy in ["independent", "cooperative", "competitive"]:
            self.coordination_strategy = strategy
            logger.info(f"Coordination strategy set to {strategy}")
        else:
            raise ValueError(f"Unknown coordination strategy: {strategy}")
    
    def get_agent_count(self) -> int:
        """Get number of agents in the system."""
        return len(self.agents)
    
    def list_agents(self) -> list:
        """List all agent IDs."""
        return list(self.agents.keys())
