"""
Objective functions and multi-objective problem definitions.
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pymoo.core.problem import Problem


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class ObjectiveFunction:
    """Definition of a single objective function."""
    name: str
    description: str
    objective_type: ObjectiveType
    weight: float = 1.0
    bounds: Optional[tuple] = None
    
    def evaluate(self, variables: Dict[str, Any]) -> float:
        """
        Evaluate the objective function.
        
        Args:
            variables: Dictionary of decision variables
            
        Returns:
            Objective value
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class LatencyObjective(ObjectiveFunction):
    """Latency minimization objective for 5G networks."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="latency",
            description="Minimize end-to-end latency",
            objective_type=ObjectiveType.MINIMIZE,
            weight=weight
        )
    
    def evaluate(self, variables: Dict[str, Any]) -> float:
        """Calculate latency based on network parameters."""
        # Simplified latency calculation
        power_allocation = variables.get("power_allocation", 1.0)
        beamforming_gain = variables.get("beamforming_gain", 1.0)
        scheduling_efficiency = variables.get("scheduling_efficiency", 1.0)
        
        # Latency decreases with better resource allocation
        latency = 10.0 / (power_allocation * beamforming_gain * scheduling_efficiency)
        return latency


class ThroughputObjective(ObjectiveFunction):
    """Throughput maximization objective for 5G networks."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="throughput",
            description="Maximize network throughput",
            objective_type=ObjectiveType.MAXIMIZE,
            weight=weight
        )
    
    def evaluate(self, variables: Dict[str, Any]) -> float:
        """Calculate throughput based on network parameters."""
        power_allocation = variables.get("power_allocation", 1.0)
        beamforming_gain = variables.get("beamforming_gain", 1.0)
        bandwidth_efficiency = variables.get("bandwidth_efficiency", 1.0)
        
        # Throughput increases with better resource utilization
        throughput = power_allocation * beamforming_gain * bandwidth_efficiency * 100.0
        
        # Return negative for minimization (pymoo minimizes by default)
        return -throughput


class EnergyObjective(ObjectiveFunction):
    """Energy consumption minimization objective for 5G networks."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="energy",
            description="Minimize energy consumption",
            objective_type=ObjectiveType.MINIMIZE,
            weight=weight
        )
    
    def evaluate(self, variables: Dict[str, Any]) -> float:
        """Calculate energy consumption based on network parameters."""
        power_allocation = variables.get("power_allocation", 1.0)
        active_antennas = variables.get("active_antennas", 1.0)
        processing_load = variables.get("processing_load", 1.0)
        
        # Energy consumption increases with resource usage
        energy = power_allocation**2 + active_antennas * 0.5 + processing_load * 0.3
        return energy


class CTRObjective(ObjectiveFunction):
    """Click-through rate maximization objective for recommendation systems."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="ctr",
            description="Maximize click-through rate",
            objective_type=ObjectiveType.MAXIMIZE,
            weight=weight
        )
    
    def evaluate(self, variables: Dict[str, Any]) -> float:
        """Calculate CTR based on recommendation parameters."""
        relevance_score = variables.get("relevance_score", 0.5)
        diversity_factor = variables.get("diversity_factor", 0.5)
        personalization = variables.get("personalization", 0.5)
        
        # CTR increases with relevance and personalization
        ctr = relevance_score * 0.6 + personalization * 0.3 + diversity_factor * 0.1
        
        # Return negative for minimization
        return -ctr


class CVRObjective(ObjectiveFunction):
    """Conversion rate maximization objective for recommendation systems."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="cvr",
            description="Maximize conversion rate",
            objective_type=ObjectiveType.MAXIMIZE,
            weight=weight
        )
    
    def evaluate(self, variables: Dict[str, Any]) -> float:
        """Calculate CVR based on recommendation parameters."""
        relevance_score = variables.get("relevance_score", 0.5)
        price_optimization = variables.get("price_optimization", 0.5)
        timing_factor = variables.get("timing_factor", 0.5)
        
        # CVR increases with relevance and price optimization
        cvr = relevance_score * 0.5 + price_optimization * 0.4 + timing_factor * 0.1
        
        # Return negative for minimization
        return -cvr


class DiversityObjective(ObjectiveFunction):
    """Diversity maximization objective for recommendation systems."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="diversity",
            description="Maximize recommendation diversity",
            objective_type=ObjectiveType.MAXIMIZE,
            weight=weight
        )
    
    def evaluate(self, variables: Dict[str, Any]) -> float:
        """Calculate diversity based on recommendation parameters."""
        category_spread = variables.get("category_spread", 0.5)
        novelty_factor = variables.get("novelty_factor", 0.5)
        exploration_rate = variables.get("exploration_rate", 0.5)
        
        # Diversity increases with category spread and novelty
        diversity = category_spread * 0.4 + novelty_factor * 0.4 + exploration_rate * 0.2
        
        # Return negative for minimization
        return -diversity


class MultiObjectiveProblem:
    """
    Multi-objective optimization problem definition.
    
    Combines multiple objective functions with decision variables
    and constraints for optimization.
    """
    
    def __init__(
        self,
        name: str,
        objectives: List[ObjectiveFunction],
        variable_names: List[str],
        variable_bounds: List[tuple],
        constraints: Optional[List[Callable]] = None
    ):
        self.name = name
        self.objectives = objectives
        self.variable_names = variable_names
        self.variable_bounds = variable_bounds
        self.constraints = constraints or []
        
        # Validate inputs
        if len(variable_names) != len(variable_bounds):
            raise ValueError("Number of variable names must match number of bounds")
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all objectives for given decision variables.
        
        Args:
            x: Decision variables array
            
        Returns:
            Array of objective values
        """
        # Convert array to variable dictionary
        variables = {name: x[i] for i, name in enumerate(self.variable_names)}
        
        # Evaluate all objectives
        objective_values = []
        for obj in self.objectives:
            value = obj.evaluate(variables)
            objective_values.append(value)
        
        return np.array(objective_values)
    
    def to_pymoo_problem(self) -> Problem:
        """Convert to pymoo Problem format."""
        
        class PymooProblem(Problem):
            def __init__(self, mo_problem):
                self.mo_problem = mo_problem
                
                # Extract bounds
                xl = [bound[0] for bound in mo_problem.variable_bounds]
                xu = [bound[1] for bound in mo_problem.variable_bounds]
                
                super().__init__(
                    n_var=len(mo_problem.variable_names),
                    n_obj=len(mo_problem.objectives),
                    n_constr=len(mo_problem.constraints),
                    xl=np.array(xl),
                    xu=np.array(xu)
                )
            
            def _evaluate(self, x, out, *args, **kwargs):
                # Evaluate objectives for each solution
                f = np.array([self.mo_problem.evaluate(xi) for xi in x])
                out["F"] = f
                
                # Evaluate constraints if any
                if self.mo_problem.constraints:
                    g = np.array([
                        [constraint(xi) for constraint in self.mo_problem.constraints]
                        for xi in x
                    ])
                    out["G"] = g
        
        return PymooProblem(self)


# Predefined problem templates
def create_5g_optimization_problem() -> MultiObjectiveProblem:
    """Create a 5G network optimization problem."""
    
    objectives = [
        LatencyObjective(),
        ThroughputObjective(),
        EnergyObjective(),
    ]
    
    variable_names = [
        "power_allocation",
        "beamforming_gain", 
        "scheduling_efficiency",
        "bandwidth_efficiency",
        "active_antennas",
        "processing_load"
    ]
    
    variable_bounds = [
        (0.1, 2.0),  # power_allocation
        (0.5, 3.0),  # beamforming_gain
        (0.1, 1.0),  # scheduling_efficiency
        (0.1, 1.0),  # bandwidth_efficiency
        (0.1, 1.0),  # active_antennas
        (0.1, 1.0),  # processing_load
    ]
    
    return MultiObjectiveProblem(
        name="5G Network Optimization",
        objectives=objectives,
        variable_names=variable_names,
        variable_bounds=variable_bounds
    )


def create_recommendation_optimization_problem() -> MultiObjectiveProblem:
    """Create a recommendation system optimization problem."""
    
    objectives = [
        CTRObjective(),
        CVRObjective(),
        DiversityObjective(),
    ]
    
    variable_names = [
        "relevance_score",
        "diversity_factor",
        "personalization",
        "price_optimization",
        "timing_factor",
        "category_spread",
        "novelty_factor",
        "exploration_rate"
    ]
    
    variable_bounds = [
        (0.0, 1.0),  # relevance_score
        (0.0, 1.0),  # diversity_factor
        (0.0, 1.0),  # personalization
        (0.0, 1.0),  # price_optimization
        (0.0, 1.0),  # timing_factor
        (0.0, 1.0),  # category_spread
        (0.0, 1.0),  # novelty_factor
        (0.0, 1.0),  # exploration_rate
    ]
    
    return MultiObjectiveProblem(
        name="Recommendation System Optimization",
        objectives=objectives,
        variable_names=variable_names,
        variable_bounds=variable_bounds
    )
