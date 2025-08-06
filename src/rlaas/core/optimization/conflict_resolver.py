"""
Conflict resolution for multi-objective optimization.

Implements various methods for resolving conflicts between objectives
and selecting the best solution from the Pareto frontier.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import euclidean

from .pareto import ParetoFrontier, ParetoSolution

logger = logging.getLogger(__name__)


class ConflictResolver(ABC):
    """Abstract base class for conflict resolution methods."""
    
    @abstractmethod
    async def resolve(
        self,
        pareto_frontier: ParetoFrontier,
        weights: Dict[str, float],
        **kwargs
    ) -> ParetoSolution:
        """
        Resolve conflicts and select the best solution.
        
        Args:
            pareto_frontier: Pareto frontier with candidate solutions
            weights: Objective weights for decision making
            **kwargs: Additional parameters
            
        Returns:
            Selected best solution
        """
        pass


class TOPSISResolver(ConflictResolver):
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) resolver.
    
    Selects the solution that is closest to the ideal solution and
    farthest from the negative ideal solution.
    """
    
    async def resolve(
        self,
        pareto_frontier: ParetoFrontier,
        weights: Dict[str, float],
        **kwargs
    ) -> ParetoSolution:
        """
        Apply TOPSIS method to select the best solution.
        
        Args:
            pareto_frontier: Pareto frontier with candidate solutions
            weights: Objective weights
            
        Returns:
            Best solution according to TOPSIS
        """
        if not pareto_frontier.solutions:
            raise ValueError("Pareto frontier is empty")
        
        if len(pareto_frontier.solutions) == 1:
            return pareto_frontier.solutions[0]
        
        logger.info(f"Applying TOPSIS to {len(pareto_frontier.solutions)} solutions")
        
        # Step 1: Create decision matrix
        decision_matrix = self._create_decision_matrix(pareto_frontier)
        objective_names = list(pareto_frontier.solutions[0].objectives.keys())
        
        # Step 2: Normalize the decision matrix
        normalized_matrix = self._normalize_matrix(decision_matrix)
        
        # Step 3: Apply weights
        weight_vector = np.array([weights.get(name, 1.0) for name in objective_names])
        weighted_matrix = normalized_matrix * weight_vector
        
        # Step 4: Determine ideal and negative ideal solutions
        ideal_solution = np.min(weighted_matrix, axis=0)  # Assuming minimization
        negative_ideal = np.max(weighted_matrix, axis=0)
        
        # Step 5: Calculate distances to ideal solutions
        distances_to_ideal = []
        distances_to_negative = []
        
        for i, row in enumerate(weighted_matrix):
            dist_ideal = euclidean(row, ideal_solution)
            dist_negative = euclidean(row, negative_ideal)
            
            distances_to_ideal.append(dist_ideal)
            distances_to_negative.append(dist_negative)
        
        # Step 6: Calculate TOPSIS scores
        topsis_scores = []
        for i in range(len(pareto_frontier.solutions)):
            if distances_to_ideal[i] + distances_to_negative[i] == 0:
                score = 0.5  # Neutral score if both distances are zero
            else:
                score = distances_to_negative[i] / (
                    distances_to_ideal[i] + distances_to_negative[i]
                )
            topsis_scores.append(score)
        
        # Step 7: Select solution with highest TOPSIS score
        best_index = np.argmax(topsis_scores)
        best_solution = pareto_frontier.solutions[best_index]
        
        # Add TOPSIS metadata
        best_solution.metadata.update({
            "topsis_score": topsis_scores[best_index],
            "distance_to_ideal": distances_to_ideal[best_index],
            "distance_to_negative": distances_to_negative[best_index],
            "selection_method": "TOPSIS"
        })
        
        logger.info(f"Selected solution {best_solution.id} with TOPSIS score {topsis_scores[best_index]:.4f}")
        
        return best_solution
    
    def _create_decision_matrix(self, pareto_frontier: ParetoFrontier) -> np.ndarray:
        """Create decision matrix from Pareto frontier."""
        objective_names = list(pareto_frontier.solutions[0].objectives.keys())
        
        matrix = []
        for solution in pareto_frontier.solutions:
            row = [solution.objectives[name] for name in objective_names]
            matrix.append(row)
        
        return np.array(matrix)
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize decision matrix using vector normalization."""
        # Calculate column norms
        norms = np.linalg.norm(matrix, axis=0)
        
        # Avoid division by zero
        norms[norms == 0] = 1.0
        
        # Normalize
        normalized = matrix / norms
        
        return normalized


class WeightedSumResolver(ConflictResolver):
    """
    Weighted sum method for conflict resolution.
    
    Selects the solution with the best weighted sum of objectives.
    """
    
    async def resolve(
        self,
        pareto_frontier: ParetoFrontier,
        weights: Dict[str, float],
        **kwargs
    ) -> ParetoSolution:
        """
        Apply weighted sum method to select the best solution.
        
        Args:
            pareto_frontier: Pareto frontier with candidate solutions
            weights: Objective weights
            
        Returns:
            Best solution according to weighted sum
        """
        if not pareto_frontier.solutions:
            raise ValueError("Pareto frontier is empty")
        
        if len(pareto_frontier.solutions) == 1:
            return pareto_frontier.solutions[0]
        
        logger.info(f"Applying weighted sum to {len(pareto_frontier.solutions)} solutions")
        
        best_solution = None
        best_score = float('inf')
        
        for solution in pareto_frontier.solutions:
            # Calculate weighted sum
            weighted_sum = 0.0
            for obj_name, obj_value in solution.objectives.items():
                weight = weights.get(obj_name, 1.0)
                weighted_sum += weight * obj_value
            
            # Update best solution (assuming minimization)
            if weighted_sum < best_score:
                best_score = weighted_sum
                best_solution = solution
        
        # Add metadata
        best_solution.metadata.update({
            "weighted_sum_score": best_score,
            "selection_method": "WeightedSum"
        })
        
        logger.info(f"Selected solution {best_solution.id} with weighted sum {best_score:.4f}")
        
        return best_solution


class CompromiseProgrammingResolver(ConflictResolver):
    """
    Compromise programming method for conflict resolution.
    
    Selects the solution that minimizes the distance to the ideal solution
    using different Lp metrics.
    """
    
    def __init__(self, p: float = 2.0):
        """
        Initialize compromise programming resolver.
        
        Args:
            p: Parameter for Lp metric (1=Manhattan, 2=Euclidean, inf=Chebyshev)
        """
        self.p = p
    
    async def resolve(
        self,
        pareto_frontier: ParetoFrontier,
        weights: Dict[str, float],
        **kwargs
    ) -> ParetoSolution:
        """
        Apply compromise programming to select the best solution.
        
        Args:
            pareto_frontier: Pareto frontier with candidate solutions
            weights: Objective weights
            
        Returns:
            Best solution according to compromise programming
        """
        if not pareto_frontier.solutions:
            raise ValueError("Pareto frontier is empty")
        
        if len(pareto_frontier.solutions) == 1:
            return pareto_frontier.solutions[0]
        
        logger.info(f"Applying compromise programming (p={self.p}) to {len(pareto_frontier.solutions)} solutions")
        
        # Find ideal solution (minimum for each objective)
        objective_names = list(pareto_frontier.solutions[0].objectives.keys())
        ideal_solution = {}
        
        for obj_name in objective_names:
            min_value = min(s.objectives[obj_name] for s in pareto_frontier.solutions)
            ideal_solution[obj_name] = min_value
        
        # Calculate compromise programming distances
        best_solution = None
        best_distance = float('inf')
        
        for solution in pareto_frontier.solutions:
            # Calculate Lp distance to ideal solution
            distance = 0.0
            
            for obj_name in objective_names:
                weight = weights.get(obj_name, 1.0)
                obj_value = solution.objectives[obj_name]
                ideal_value = ideal_solution[obj_name]
                
                # Normalize by ideal value to avoid scale issues
                if ideal_value != 0:
                    normalized_diff = abs(obj_value - ideal_value) / abs(ideal_value)
                else:
                    normalized_diff = abs(obj_value - ideal_value)
                
                if self.p == float('inf'):
                    # Chebyshev distance (max norm)
                    distance = max(distance, weight * normalized_diff)
                else:
                    # Lp norm
                    distance += (weight * normalized_diff) ** self.p
            
            if self.p != float('inf'):
                distance = distance ** (1.0 / self.p)
            
            # Update best solution
            if distance < best_distance:
                best_distance = distance
                best_solution = solution
        
        # Add metadata
        best_solution.metadata.update({
            "compromise_distance": best_distance,
            "p_parameter": self.p,
            "selection_method": "CompromiseProgramming"
        })
        
        logger.info(f"Selected solution {best_solution.id} with compromise distance {best_distance:.4f}")
        
        return best_solution


class AdaptiveResolver(ConflictResolver):
    """
    Adaptive conflict resolver that selects the best method based on context.
    
    Chooses between different resolution methods based on the problem
    characteristics and user preferences.
    """
    
    def __init__(self):
        self.topsis = TOPSISResolver()
        self.weighted_sum = WeightedSumResolver()
        self.compromise = CompromiseProgrammingResolver()
    
    async def resolve(
        self,
        pareto_frontier: ParetoFrontier,
        weights: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ParetoSolution:
        """
        Adaptively select and apply the best resolution method.
        
        Args:
            pareto_frontier: Pareto frontier with candidate solutions
            weights: Objective weights
            context: Additional context for method selection
            
        Returns:
            Best solution according to adaptive selection
        """
        context = context or {}
        
        # Select method based on context
        method = self._select_method(pareto_frontier, weights, context)
        
        logger.info(f"Adaptively selected {method.__class__.__name__}")
        
        # Apply selected method
        return await method.resolve(pareto_frontier, weights, **kwargs)
    
    def _select_method(
        self,
        pareto_frontier: ParetoFrontier,
        weights: Dict[str, float],
        context: Dict[str, Any]
    ) -> ConflictResolver:
        """Select the best resolution method based on context."""
        
        n_solutions = len(pareto_frontier.solutions)
        n_objectives = len(pareto_frontier.solutions[0].objectives) if n_solutions > 0 else 0
        
        # Use TOPSIS for balanced multi-objective problems
        if n_objectives >= 3 and n_solutions >= 5:
            return self.topsis
        
        # Use weighted sum for simple problems or when weights are very uneven
        weight_values = list(weights.values())
        if len(weight_values) > 0:
            weight_ratio = max(weight_values) / min(weight_values) if min(weight_values) > 0 else float('inf')
            if weight_ratio > 5.0:  # Very uneven weights
                return self.weighted_sum
        
        # Use compromise programming for emergency scenarios
        if context.get("mode") == "emergency":
            return self.compromise
        
        # Default to TOPSIS
        return self.topsis
