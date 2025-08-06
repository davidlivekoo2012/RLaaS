"""
Main optimization engine for multi-objective optimization.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .objectives import MultiObjectiveProblem, ObjectiveFunction
from .conflict_resolver import ConflictResolver, TOPSISResolver
from .pareto import ParetoFrontier, ParetoSolution
from rlaas.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class OptimizationAlgorithm(Enum):
    """Supported optimization algorithms."""
    NSGA3 = "nsga3"
    MOEAD = "moead"


class OptimizationMode(Enum):
    """Optimization modes for different scenarios."""
    NORMAL = "normal"
    EMERGENCY = "emergency"
    REVENUE_FOCUSED = "revenue_focused"
    USER_EXPERIENCE = "user_experience"


@dataclass
class OptimizationRequest:
    """Optimization request configuration."""
    problem: MultiObjectiveProblem
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA3
    mode: OptimizationMode = OptimizationMode.NORMAL
    population_size: int = 100
    generations: int = 500
    weights: Optional[Dict[str, float]] = None
    constraints: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None


@dataclass
class OptimizationResult:
    """Optimization result."""
    pareto_frontier: ParetoFrontier
    best_solution: ParetoSolution
    convergence_history: List[float]
    execution_time: float
    algorithm_used: OptimizationAlgorithm
    mode_used: OptimizationMode
    metadata: Dict[str, Any]


class OptimizationEngine:
    """
    Main optimization engine for multi-objective optimization.
    
    Supports NSGA-III and MOEA/D algorithms with dynamic weight adjustment
    and conflict resolution for 5G networks and recommendation systems.
    """
    
    def __init__(self):
        self.conflict_resolver = ConflictResolver()
        self.topsis_resolver = TOPSISResolver()
        self._running_optimizations: Dict[str, asyncio.Task] = {}
    
    async def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Run multi-objective optimization.
        
        Args:
            request: Optimization request configuration
            
        Returns:
            Optimization result with Pareto frontier and best solution
        """
        logger.info(f"Starting optimization with {request.algorithm.value} algorithm")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get algorithm-specific weights
            weights = self._get_weights_for_mode(request.mode, request.problem)
            if request.weights:
                weights.update(request.weights)
            
            # Create algorithm instance
            algorithm = self._create_algorithm(
                request.algorithm,
                request.problem,
                request.population_size
            )
            
            # Run optimization
            result = await self._run_optimization(
                algorithm,
                request.problem,
                request.generations,
                request.timeout
            )
            
            # Create Pareto frontier
            pareto_frontier = self._create_pareto_frontier(result, request.problem)
            
            # Resolve conflicts and select best solution
            best_solution = await self._resolve_conflicts(
                pareto_frontier,
                weights,
                request.mode
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Optimization completed in {execution_time:.2f} seconds")
            
            return OptimizationResult(
                pareto_frontier=pareto_frontier,
                best_solution=best_solution,
                convergence_history=self._extract_convergence_history(result),
                execution_time=execution_time,
                algorithm_used=request.algorithm,
                mode_used=request.mode,
                metadata={
                    "population_size": request.population_size,
                    "generations": request.generations,
                    "weights": weights,
                    "n_solutions": len(pareto_frontier.solutions),
                }
            )
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _get_weights_for_mode(
        self, 
        mode: OptimizationMode, 
        problem: MultiObjectiveProblem
    ) -> Dict[str, float]:
        """Get optimization weights based on mode and problem type."""
        
        if "5g" in problem.name.lower() or "network" in problem.name.lower():
            return self._get_5g_weights(mode)
        elif "recommendation" in problem.name.lower() or "rec" in problem.name.lower():
            return self._get_recommendation_weights(mode)
        else:
            # Default equal weights
            n_objectives = len(problem.objectives)
            return {obj.name: 1.0 / n_objectives for obj in problem.objectives}
    
    def _get_5g_weights(self, mode: OptimizationMode) -> Dict[str, float]:
        """Get weights for 5G network optimization."""
        if mode == OptimizationMode.EMERGENCY:
            return {
                "latency": 0.6,
                "throughput": 0.2,
                "energy": 0.1,
                "satisfaction": 0.1,
            }
        else:  # NORMAL mode
            return {
                "latency": 0.25,
                "throughput": 0.25,
                "energy": 0.25,
                "satisfaction": 0.25,
            }
    
    def _get_recommendation_weights(self, mode: OptimizationMode) -> Dict[str, float]:
        """Get weights for recommendation system optimization."""
        if mode == OptimizationMode.REVENUE_FOCUSED:
            return {
                "ctr": 0.1,
                "cvr": 0.6,
                "diversity": 0.1,
                "cost": 0.2,
            }
        else:  # USER_EXPERIENCE mode
            return {
                "ctr": 0.5,
                "cvr": 0.2,
                "diversity": 0.2,
                "cost": 0.1,
            }
    
    def _create_algorithm(
        self,
        algorithm_type: OptimizationAlgorithm,
        problem: MultiObjectiveProblem,
        population_size: int
    ):
        """Create optimization algorithm instance."""
        
        if algorithm_type == OptimizationAlgorithm.NSGA3:
            # Create reference directions for NSGA-III
            ref_dirs = get_reference_directions(
                "das-dennis",
                len(problem.objectives),
                n_partitions=12
            )
            return NSGA3(
                pop_size=population_size,
                ref_dirs=ref_dirs
            )
        
        elif algorithm_type == OptimizationAlgorithm.MOEAD:
            # Create reference directions for MOEA/D
            ref_dirs = get_reference_directions(
                "das-dennis",
                len(problem.objectives),
                n_partitions=12
            )
            return MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=20,
                decomposition="tchebycheff",
                prob_neighbor_mating=0.7
            )
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_type}")
    
    async def _run_optimization(
        self,
        algorithm,
        problem: MultiObjectiveProblem,
        generations: int,
        timeout: Optional[int] = None
    ):
        """Run the optimization algorithm asynchronously."""
        
        def run_sync():
            return minimize(
                problem.to_pymoo_problem(),
                algorithm,
                ("n_gen", generations),
                verbose=True
            )
        
        # Run optimization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        if timeout:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_sync),
                timeout=timeout
            )
        else:
            result = await loop.run_in_executor(None, run_sync)
        
        return result
    
    def _create_pareto_frontier(
        self,
        result,
        problem: MultiObjectiveProblem
    ) -> ParetoFrontier:
        """Create Pareto frontier from optimization result."""
        
        solutions = []
        
        for i, (x, f) in enumerate(zip(result.X, result.F)):
            # Create objective values dictionary
            objectives = {}
            for j, obj in enumerate(problem.objectives):
                objectives[obj.name] = f[j]
            
            # Create decision variables dictionary
            variables = {}
            for j, var_name in enumerate(problem.variable_names):
                variables[var_name] = x[j]
            
            solution = ParetoSolution(
                id=f"sol_{i}",
                objectives=objectives,
                variables=variables,
                rank=getattr(result, 'rank', [0] * len(result.X))[i],
                crowding_distance=getattr(result, 'crowding', [0] * len(result.X))[i]
            )
            solutions.append(solution)
        
        return ParetoFrontier(solutions=solutions)
    
    async def _resolve_conflicts(
        self,
        pareto_frontier: ParetoFrontier,
        weights: Dict[str, float],
        mode: OptimizationMode
    ) -> ParetoSolution:
        """Resolve conflicts and select the best solution."""
        
        # Use TOPSIS method for conflict resolution
        best_solution = await self.topsis_resolver.resolve(
            pareto_frontier,
            weights
        )
        
        logger.info(f"Selected best solution: {best_solution.id}")
        return best_solution
    
    def _extract_convergence_history(self, result) -> List[float]:
        """Extract convergence history from optimization result."""
        
        # Extract hypervolume or other convergence metrics if available
        if hasattr(result, 'history'):
            return [entry.get('hv', 0.0) for entry in result.history]
        else:
            # Return empty list if no history available
            return []
    
    async def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """Get status of running optimization."""
        
        if optimization_id not in self._running_optimizations:
            return {"status": "not_found"}
        
        task = self._running_optimizations[optimization_id]
        
        if task.done():
            if task.exception():
                return {
                    "status": "failed",
                    "error": str(task.exception())
                }
            else:
                return {
                    "status": "completed",
                    "result": task.result()
                }
        else:
            return {"status": "running"}
    
    async def cancel_optimization(self, optimization_id: str) -> bool:
        """Cancel a running optimization."""
        
        if optimization_id not in self._running_optimizations:
            return False
        
        task = self._running_optimizations[optimization_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self._running_optimizations[optimization_id]
        return True
