"""
RL environments for 5G networks and recommendation systems.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import gym
from gym import spaces

logger = logging.getLogger(__name__)


class NetworkEnvironment(gym.Env):
    """
    5G Network optimization environment.
    
    State space includes network KPIs, user distribution, and interference levels.
    Action space includes power allocation, beamforming, and scheduling parameters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        
        # Network parameters
        self.n_cells = self.config.get("n_cells", 10)
        self.n_users = self.config.get("n_users", 100)
        self.max_power = self.config.get("max_power", 10.0)  # Watts
        self.bandwidth = self.config.get("bandwidth", 100.0)  # MHz
        
        # State space: [latency, throughput, energy, user_satisfaction, interference, load]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6 * self.n_cells,),
            dtype=np.float32
        )
        
        # Action space: [power_allocation, beamforming_weights, scheduling_priority]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 * self.n_cells,),
            dtype=np.float32
        )
        
        # Environment state
        self.current_state = None
        self.step_count = 0
        self.max_steps = self.config.get("max_steps", 1000)
        
        # Performance targets
        self.target_latency = 1.0  # ms
        self.target_throughput = 1000.0  # Mbps
        self.target_energy_efficiency = 0.8
        self.target_satisfaction = 0.95
        
        logger.info(f"NetworkEnvironment initialized with {self.n_cells} cells")
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.step_count = 0
        
        # Initialize random network state
        self.current_state = np.random.uniform(0.3, 0.7, self.observation_space.shape)
        
        return self.current_state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        
        # Parse actions
        power_actions = action[:self.n_cells]
        beamforming_actions = action[self.n_cells:2*self.n_cells]
        scheduling_actions = action[2*self.n_cells:]
        
        # Simulate network dynamics
        new_state = self._simulate_network_step(
            power_actions, beamforming_actions, scheduling_actions
        )
        
        # Calculate reward
        reward = self._calculate_reward(new_state, action)
        
        # Update state
        self.current_state = new_state
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # Additional info
        info = {
            "latency": np.mean(new_state[::6]),
            "throughput": np.mean(new_state[1::6]),
            "energy": np.mean(new_state[2::6]),
            "satisfaction": np.mean(new_state[3::6]),
            "interference": np.mean(new_state[4::6]),
            "load": np.mean(new_state[5::6]),
            "step": self.step_count
        }
        
        return new_state.copy(), reward, done, info
    
    def _simulate_network_step(
        self,
        power_actions: np.ndarray,
        beamforming_actions: np.ndarray,
        scheduling_actions: np.ndarray
    ) -> np.ndarray:
        """Simulate one step of network dynamics."""
        
        new_state = self.current_state.copy()
        
        for cell_idx in range(self.n_cells):
            base_idx = cell_idx * 6
            
            # Current cell state
            latency = self.current_state[base_idx]
            throughput = self.current_state[base_idx + 1]
            energy = self.current_state[base_idx + 2]
            satisfaction = self.current_state[base_idx + 3]
            interference = self.current_state[base_idx + 4]
            load = self.current_state[base_idx + 5]
            
            # Action effects
            power_factor = power_actions[cell_idx]
            beamforming_factor = beamforming_actions[cell_idx]
            scheduling_factor = scheduling_actions[cell_idx]
            
            # Update latency (lower is better)
            latency_improvement = (beamforming_factor + scheduling_factor) * 0.1
            new_latency = max(0.01, latency - latency_improvement + np.random.normal(0, 0.02))
            
            # Update throughput (higher is better)
            throughput_improvement = (power_factor + beamforming_factor) * 0.15
            new_throughput = min(1.0, throughput + throughput_improvement + np.random.normal(0, 0.02))
            
            # Update energy consumption (lower is better)
            energy_increase = power_factor * 0.2
            new_energy = min(1.0, energy + energy_increase + np.random.normal(0, 0.01))
            
            # Update user satisfaction
            satisfaction_change = (new_throughput - throughput) - (new_latency - latency)
            new_satisfaction = np.clip(satisfaction + satisfaction_change * 0.1, 0.0, 1.0)
            
            # Update interference (affected by power)
            interference_change = (power_factor - 0.5) * 0.1
            new_interference = np.clip(interference + interference_change, 0.0, 1.0)
            
            # Update load (random walk)
            new_load = np.clip(load + np.random.normal(0, 0.05), 0.1, 1.0)
            
            # Store new state
            new_state[base_idx] = new_latency
            new_state[base_idx + 1] = new_throughput
            new_state[base_idx + 2] = new_energy
            new_state[base_idx + 3] = new_satisfaction
            new_state[base_idx + 4] = new_interference
            new_state[base_idx + 5] = new_load
        
        return new_state
    
    def _calculate_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward based on network performance."""
        
        # Extract metrics
        latency = np.mean(state[::6])
        throughput = np.mean(state[1::6])
        energy = np.mean(state[2::6])
        satisfaction = np.mean(state[3::6])
        
        # Multi-objective reward
        latency_reward = max(0, 1.0 - latency)  # Lower latency is better
        throughput_reward = throughput  # Higher throughput is better
        energy_reward = max(0, 1.0 - energy)  # Lower energy is better
        satisfaction_reward = satisfaction  # Higher satisfaction is better
        
        # Weighted combination
        reward = (
            0.3 * latency_reward +
            0.3 * throughput_reward +
            0.2 * energy_reward +
            0.2 * satisfaction_reward
        )
        
        # Penalty for extreme actions
        action_penalty = np.sum(np.abs(action - 0.5)) * 0.01
        reward -= action_penalty
        
        return reward
    
    def render(self, mode: str = "human"):
        """Render the environment (optional)."""
        if mode == "human":
            print(f"Step: {self.step_count}")
            print(f"Avg Latency: {np.mean(self.current_state[::6]):.3f}")
            print(f"Avg Throughput: {np.mean(self.current_state[1::6]):.3f}")
            print(f"Avg Energy: {np.mean(self.current_state[2::6]):.3f}")
            print(f"Avg Satisfaction: {np.mean(self.current_state[3::6]):.3f}")


class RecommendationEnvironment(gym.Env):
    """
    Recommendation system optimization environment.
    
    State space includes user profiles, item features, and context.
    Action space includes recommendation strategies and parameters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        
        # System parameters
        self.n_users = self.config.get("n_users", 1000)
        self.n_items = self.config.get("n_items", 10000)
        self.n_categories = self.config.get("n_categories", 20)
        self.recommendation_size = self.config.get("recommendation_size", 10)
        
        # State space: [user_features, item_features, context, system_metrics]
        state_dim = 50  # Simplified feature representation
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: [relevance_weight, diversity_weight, novelty_weight, exploration_rate]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Environment state
        self.current_state = None
        self.step_count = 0
        self.max_steps = self.config.get("max_steps", 1000)
        
        # Performance metrics
        self.ctr_history = []
        self.cvr_history = []
        self.diversity_history = []
        
        logger.info(f"RecommendationEnvironment initialized")
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.step_count = 0
        self.ctr_history = []
        self.cvr_history = []
        self.diversity_history = []
        
        # Initialize random state
        self.current_state = np.random.uniform(0.2, 0.8, self.observation_space.shape)
        
        return self.current_state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        
        # Parse actions
        relevance_weight = action[0]
        diversity_weight = action[1]
        novelty_weight = action[2]
        exploration_rate = action[3]
        
        # Simulate recommendation system step
        new_state, metrics = self._simulate_recommendation_step(
            relevance_weight, diversity_weight, novelty_weight, exploration_rate
        )
        
        # Calculate reward
        reward = self._calculate_reward(metrics, action)
        
        # Update state and history
        self.current_state = new_state
        self.step_count += 1
        self.ctr_history.append(metrics["ctr"])
        self.cvr_history.append(metrics["cvr"])
        self.diversity_history.append(metrics["diversity"])
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # Additional info
        info = {
            "ctr": metrics["ctr"],
            "cvr": metrics["cvr"],
            "diversity": metrics["diversity"],
            "cost": metrics["cost"],
            "step": self.step_count,
            "avg_ctr": np.mean(self.ctr_history),
            "avg_cvr": np.mean(self.cvr_history),
            "avg_diversity": np.mean(self.diversity_history)
        }
        
        return new_state.copy(), reward, done, info
    
    def _simulate_recommendation_step(
        self,
        relevance_weight: float,
        diversity_weight: float,
        novelty_weight: float,
        exploration_rate: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Simulate one step of recommendation system."""
        
        # Simulate user interactions based on strategy
        base_ctr = 0.1
        base_cvr = 0.05
        base_diversity = 0.3
        base_cost = 0.5
        
        # CTR influenced by relevance and exploration
        ctr = base_ctr + relevance_weight * 0.15 + exploration_rate * 0.05
        ctr += np.random.normal(0, 0.02)
        ctr = np.clip(ctr, 0.01, 0.5)
        
        # CVR influenced by relevance but reduced by too much diversity
        cvr = base_cvr + relevance_weight * 0.08 - diversity_weight * 0.02
        cvr += np.random.normal(0, 0.01)
        cvr = np.clip(cvr, 0.005, 0.2)
        
        # Diversity influenced by diversity weight and novelty
        diversity = base_diversity + diversity_weight * 0.4 + novelty_weight * 0.2
        diversity += np.random.normal(0, 0.03)
        diversity = np.clip(diversity, 0.1, 0.9)
        
        # Cost influenced by exploration and novelty
        cost = base_cost + exploration_rate * 0.2 + novelty_weight * 0.1
        cost += np.random.normal(0, 0.02)
        cost = np.clip(cost, 0.1, 1.0)
        
        # Update state based on performance
        new_state = self.current_state.copy()
        
        # Update user engagement features
        new_state[:10] = np.clip(
            new_state[:10] + (ctr - 0.1) * 0.1 + np.random.normal(0, 0.01, 10),
            0.0, 1.0
        )
        
        # Update item popularity features
        new_state[10:20] = np.clip(
            new_state[10:20] + (cvr - 0.05) * 0.1 + np.random.normal(0, 0.01, 10),
            0.0, 1.0
        )
        
        # Update diversity features
        new_state[20:30] = np.clip(
            new_state[20:30] + (diversity - 0.3) * 0.1 + np.random.normal(0, 0.01, 10),
            0.0, 1.0
        )
        
        # Update system metrics
        new_state[30:] = np.clip(
            new_state[30:] + np.random.normal(0, 0.02, len(new_state[30:])),
            0.0, 1.0
        )
        
        metrics = {
            "ctr": ctr,
            "cvr": cvr,
            "diversity": diversity,
            "cost": cost
        }
        
        return new_state, metrics
    
    def _calculate_reward(self, metrics: Dict[str, float], action: np.ndarray) -> float:
        """Calculate reward based on recommendation performance."""
        
        ctr = metrics["ctr"]
        cvr = metrics["cvr"]
        diversity = metrics["diversity"]
        cost = metrics["cost"]
        
        # Multi-objective reward
        ctr_reward = ctr * 10  # Scale CTR
        cvr_reward = cvr * 20  # Scale CVR
        diversity_reward = diversity * 2  # Scale diversity
        cost_penalty = cost * 1  # Cost penalty
        
        # Weighted combination (can be adjusted based on business priorities)
        reward = (
            0.3 * ctr_reward +
            0.4 * cvr_reward +
            0.2 * diversity_reward -
            0.1 * cost_penalty
        )
        
        # Bonus for balanced performance
        if ctr > 0.12 and cvr > 0.06 and diversity > 0.4:
            reward += 0.5
        
        return reward
    
    def render(self, mode: str = "human"):
        """Render the environment (optional)."""
        if mode == "human":
            print(f"Step: {self.step_count}")
            if self.ctr_history:
                print(f"CTR: {self.ctr_history[-1]:.4f}")
                print(f"CVR: {self.cvr_history[-1]:.4f}")
                print(f"Diversity: {self.diversity_history[-1]:.4f}")
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        if not self.ctr_history:
            return {}
        
        return {
            "avg_ctr": np.mean(self.ctr_history),
            "avg_cvr": np.mean(self.cvr_history),
            "avg_diversity": np.mean(self.diversity_history),
            "ctr_trend": np.mean(self.ctr_history[-10:]) - np.mean(self.ctr_history[:10]) if len(self.ctr_history) >= 20 else 0,
            "cvr_trend": np.mean(self.cvr_history[-10:]) - np.mean(self.cvr_history[:10]) if len(self.cvr_history) >= 20 else 0,
        }
