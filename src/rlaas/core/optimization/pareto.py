"""
Pareto frontier and solution management.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import json


@dataclass
class ParetoSolution:
    """
    A single solution in the Pareto frontier.
    
    Represents a non-dominated solution with its objective values,
    decision variables, and additional metadata.
    """
    id: str
    objectives: Dict[str, float]
    variables: Dict[str, Any]
    rank: int = 0
    crowding_distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def dominates(self, other: 'ParetoSolution') -> bool:
        """
        Check if this solution dominates another solution.
        
        A solution dominates another if it is at least as good in all
        objectives and strictly better in at least one objective.
        
        Args:
            other: Another Pareto solution
            
        Returns:
            True if this solution dominates the other
        """
        if set(self.objectives.keys()) != set(other.objectives.keys()):
            raise ValueError("Solutions must have the same objectives")
        
        at_least_as_good = True
        strictly_better = False
        
        for obj_name in self.objectives:
            self_value = self.objectives[obj_name]
            other_value = other.objectives[obj_name]
            
            # Assuming minimization (negative values for maximization objectives)
            if self_value > other_value:
                at_least_as_good = False
                break
            elif self_value < other_value:
                strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def distance_to(self, other: 'ParetoSolution') -> float:
        """
        Calculate Euclidean distance to another solution in objective space.
        
        Args:
            other: Another Pareto solution
            
        Returns:
            Euclidean distance between solutions
        """
        if set(self.objectives.keys()) != set(other.objectives.keys()):
            raise ValueError("Solutions must have the same objectives")
        
        squared_diff = 0.0
        for obj_name in self.objectives:
            diff = self.objectives[obj_name] - other.objectives[obj_name]
            squared_diff += diff ** 2
        
        return np.sqrt(squared_diff)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary format."""
        return {
            "id": self.id,
            "objectives": self.objectives,
            "variables": self.variables,
            "rank": self.rank,
            "crowding_distance": self.crowding_distance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParetoSolution':
        """Create solution from dictionary format."""
        return cls(
            id=data["id"],
            objectives=data["objectives"],
            variables=data["variables"],
            rank=data.get("rank", 0),
            crowding_distance=data.get("crowding_distance", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class ParetoFrontier:
    """
    Collection of non-dominated solutions forming the Pareto frontier.
    
    Manages a set of Pareto-optimal solutions and provides methods
    for analysis, filtering, and selection.
    """
    solutions: List[ParetoSolution]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process solutions after initialization."""
        if not self.solutions:
            return
        
        # Ensure all solutions have the same objectives
        first_objectives = set(self.solutions[0].objectives.keys())
        for solution in self.solutions[1:]:
            if set(solution.objectives.keys()) != first_objectives:
                raise ValueError("All solutions must have the same objectives")
    
    def add_solution(self, solution: ParetoSolution) -> bool:
        """
        Add a solution to the frontier if it's non-dominated.
        
        Args:
            solution: Solution to add
            
        Returns:
            True if solution was added, False if dominated
        """
        # Check if solution is dominated by any existing solution
        for existing in self.solutions:
            if existing.dominates(solution):
                return False
        
        # Remove any solutions dominated by the new solution
        self.solutions = [
            s for s in self.solutions 
            if not solution.dominates(s)
        ]
        
        # Add the new solution
        self.solutions.append(solution)
        return True
    
    def get_extreme_solutions(self) -> Dict[str, ParetoSolution]:
        """
        Get extreme solutions for each objective.
        
        Returns:
            Dictionary mapping objective names to best solutions
        """
        if not self.solutions:
            return {}
        
        extreme_solutions = {}
        objective_names = list(self.solutions[0].objectives.keys())
        
        for obj_name in objective_names:
            best_solution = min(
                self.solutions,
                key=lambda s: s.objectives[obj_name]
            )
            extreme_solutions[obj_name] = best_solution
        
        return extreme_solutions
    
    def get_knee_solutions(self, n: int = 3) -> List[ParetoSolution]:
        """
        Get knee solutions (solutions with good trade-offs).
        
        Args:
            n: Number of knee solutions to return
            
        Returns:
            List of knee solutions
        """
        if len(self.solutions) <= n:
            return self.solutions.copy()
        
        # Sort by crowding distance (higher is better for diversity)
        sorted_solutions = sorted(
            self.solutions,
            key=lambda s: s.crowding_distance,
            reverse=True
        )
        
        return sorted_solutions[:n]
    
    def filter_by_constraints(
        self, 
        constraints: Dict[str, tuple]
    ) -> 'ParetoFrontier':
        """
        Filter solutions by objective value constraints.
        
        Args:
            constraints: Dictionary mapping objective names to (min, max) tuples
            
        Returns:
            New ParetoFrontier with filtered solutions
        """
        filtered_solutions = []
        
        for solution in self.solutions:
            satisfies_constraints = True
            
            for obj_name, (min_val, max_val) in constraints.items():
                if obj_name in solution.objectives:
                    obj_value = solution.objectives[obj_name]
                    if obj_value < min_val or obj_value > max_val:
                        satisfies_constraints = False
                        break
            
            if satisfies_constraints:
                filtered_solutions.append(solution)
        
        return ParetoFrontier(
            solutions=filtered_solutions,
            metadata={**self.metadata, "filtered_by": constraints}
        )
    
    def calculate_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """
        Calculate hypervolume indicator for the frontier.
        
        Args:
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        if not self.solutions:
            return 0.0
        
        # Simple hypervolume calculation (for 2D/3D cases)
        # For production, use specialized libraries like pygmo
        
        objective_names = list(self.solutions[0].objectives.keys())
        if len(objective_names) > 3:
            # For high-dimensional cases, return approximation
            return len(self.solutions) * 0.1
        
        # Calculate dominated volume for each solution
        total_volume = 0.0
        
        for solution in self.solutions:
            volume = 1.0
            for obj_name in objective_names:
                obj_value = solution.objectives[obj_name]
                ref_value = reference_point[obj_name]
                
                # Volume contribution (assuming minimization)
                if obj_value < ref_value:
                    volume *= (ref_value - obj_value)
                else:
                    volume = 0.0
                    break
            
            total_volume += volume
        
        return total_volume
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of the frontier.
        
        Returns:
            Dictionary with frontier statistics
        """
        if not self.solutions:
            return {"n_solutions": 0}
        
        objective_names = list(self.solutions[0].objectives.keys())
        stats = {
            "n_solutions": len(self.solutions),
            "objectives": {}
        }
        
        for obj_name in objective_names:
            values = [s.objectives[obj_name] for s in self.solutions]
            stats["objectives"][obj_name] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values)
            }
        
        # Add rank statistics
        ranks = [s.rank for s in self.solutions]
        stats["ranks"] = {
            "min": min(ranks),
            "max": max(ranks),
            "mean": np.mean(ranks)
        }
        
        # Add crowding distance statistics
        distances = [s.crowding_distance for s in self.solutions]
        stats["crowding_distances"] = {
            "min": min(distances),
            "max": max(distances),
            "mean": np.mean(distances)
        }
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert frontier to dictionary format."""
        return {
            "solutions": [s.to_dict() for s in self.solutions],
            "metadata": self.metadata,
            "statistics": self.get_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParetoFrontier':
        """Create frontier from dictionary format."""
        solutions = [
            ParetoSolution.from_dict(s_data) 
            for s_data in data["solutions"]
        ]
        return cls(
            solutions=solutions,
            metadata=data.get("metadata", {})
        )
    
    def save_to_file(self, filepath: str):
        """Save frontier to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ParetoFrontier':
        """Load frontier from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
