"""
Optimization API routes for RLaaS platform.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import logging

from rlaas.core.optimization import (
    OptimizationEngine,
    OptimizationRequest,
    OptimizationAlgorithm,
    OptimizationMode,
    create_5g_optimization_problem,
    create_recommendation_optimization_problem,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global optimization engine instance
optimization_engine = OptimizationEngine()


class OptimizationRequestModel(BaseModel):
    """API model for optimization requests."""
    problem_type: str = Field(..., description="Type of optimization problem (5g, recommendation)")
    algorithm: str = Field(default="nsga3", description="Optimization algorithm (nsga3, moead)")
    mode: str = Field(default="normal", description="Optimization mode")
    population_size: int = Field(default=100, ge=10, le=1000, description="Population size")
    generations: int = Field(default=500, ge=10, le=2000, description="Number of generations")
    weights: Optional[Dict[str, float]] = Field(default=None, description="Objective weights")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Problem constraints")
    timeout: Optional[int] = Field(default=None, ge=10, le=3600, description="Timeout in seconds")


class OptimizationResponseModel(BaseModel):
    """API model for optimization responses."""
    optimization_id: str
    status: str
    message: str


class OptimizationResultModel(BaseModel):
    """API model for optimization results."""
    optimization_id: str
    status: str
    pareto_frontier: Optional[Dict[str, Any]] = None
    best_solution: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    algorithm_used: Optional[str] = None
    mode_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ProblemTemplateModel(BaseModel):
    """API model for problem templates."""
    name: str
    description: str
    objectives: List[str]
    variables: List[str]
    use_cases: List[str]


@router.post("/optimize", response_model=OptimizationResponseModel)
async def start_optimization(
    request: OptimizationRequestModel,
    background_tasks: BackgroundTasks
) -> OptimizationResponseModel:
    """
    Start a new multi-objective optimization.
    
    Supports 5G network optimization and recommendation system optimization
    with various algorithms and modes.
    """
    try:
        # Generate optimization ID
        optimization_id = str(uuid.uuid4())
        
        # Create problem based on type
        if request.problem_type.lower() == "5g":
            problem = create_5g_optimization_problem()
        elif request.problem_type.lower() == "recommendation":
            problem = create_recommendation_optimization_problem()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported problem type: {request.problem_type}"
            )
        
        # Parse algorithm
        try:
            algorithm = OptimizationAlgorithm(request.algorithm.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}"
            )
        
        # Parse mode
        try:
            mode = OptimizationMode(request.mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported mode: {request.mode}"
            )
        
        # Create optimization request
        opt_request = OptimizationRequest(
            problem=problem,
            algorithm=algorithm,
            mode=mode,
            population_size=request.population_size,
            generations=request.generations,
            weights=request.weights,
            constraints=request.constraints,
            timeout=request.timeout
        )
        
        # Start optimization in background
        background_tasks.add_task(
            run_optimization_task,
            optimization_id,
            opt_request
        )
        
        logger.info(f"Started optimization {optimization_id}")
        
        return OptimizationResponseModel(
            optimization_id=optimization_id,
            status="started",
            message="Optimization started successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start optimization"
        )


@router.get("/optimize/{optimization_id}", response_model=OptimizationResultModel)
async def get_optimization_result(optimization_id: str) -> OptimizationResultModel:
    """
    Get the result of an optimization.
    
    Returns the current status and results if completed.
    """
    try:
        status_info = await optimization_engine.get_optimization_status(optimization_id)
        
        if status_info["status"] == "not_found":
            raise HTTPException(
                status_code=404,
                detail="Optimization not found"
            )
        
        elif status_info["status"] == "running":
            return OptimizationResultModel(
                optimization_id=optimization_id,
                status="running"
            )
        
        elif status_info["status"] == "failed":
            return OptimizationResultModel(
                optimization_id=optimization_id,
                status="failed",
                error=status_info.get("error", "Unknown error")
            )
        
        elif status_info["status"] == "completed":
            result = status_info["result"]
            
            return OptimizationResultModel(
                optimization_id=optimization_id,
                status="completed",
                pareto_frontier=result.pareto_frontier.to_dict(),
                best_solution=result.best_solution.to_dict(),
                execution_time=result.execution_time,
                algorithm_used=result.algorithm_used.value,
                mode_used=result.mode_used.value,
                metadata=result.metadata
            )
        
        else:
            raise HTTPException(
                status_code=500,
                detail="Unknown optimization status"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization result: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get optimization result"
        )


@router.delete("/optimize/{optimization_id}")
async def cancel_optimization(optimization_id: str) -> Dict[str, str]:
    """
    Cancel a running optimization.
    """
    try:
        success = await optimization_engine.cancel_optimization(optimization_id)
        
        if success:
            return {"message": "Optimization cancelled successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail="Optimization not found or already completed"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel optimization"
        )


@router.get("/templates", response_model=List[ProblemTemplateModel])
async def get_problem_templates() -> List[ProblemTemplateModel]:
    """
    Get available optimization problem templates.
    """
    templates = [
        ProblemTemplateModel(
            name="5G Network Optimization",
            description="Multi-objective optimization for 5G network parameters",
            objectives=["latency", "throughput", "energy"],
            variables=[
                "power_allocation",
                "beamforming_gain",
                "scheduling_efficiency",
                "bandwidth_efficiency",
                "active_antennas",
                "processing_load"
            ],
            use_cases=[
                "Network slicing optimization",
                "Resource allocation",
                "Energy efficiency",
                "QoS optimization"
            ]
        ),
        ProblemTemplateModel(
            name="Recommendation System Optimization",
            description="Multi-objective optimization for recommendation systems",
            objectives=["ctr", "cvr", "diversity"],
            variables=[
                "relevance_score",
                "diversity_factor",
                "personalization",
                "price_optimization",
                "timing_factor",
                "category_spread",
                "novelty_factor",
                "exploration_rate"
            ],
            use_cases=[
                "E-commerce recommendations",
                "Content recommendations",
                "Ad targeting",
                "Cold start optimization"
            ]
        )
    ]
    
    return templates


@router.get("/algorithms")
async def get_supported_algorithms() -> Dict[str, Any]:
    """
    Get information about supported optimization algorithms.
    """
    return {
        "algorithms": [
            {
                "name": "nsga3",
                "description": "NSGA-III - Non-dominated Sorting Genetic Algorithm III",
                "best_for": "Many-objective optimization (3+ objectives)",
                "parameters": ["population_size", "generations"]
            },
            {
                "name": "moead",
                "description": "MOEA/D - Multi-Objective Evolutionary Algorithm based on Decomposition",
                "best_for": "Decomposition-based multi-objective optimization",
                "parameters": ["population_size", "generations", "n_neighbors"]
            }
        ],
        "modes": [
            {
                "name": "normal",
                "description": "Balanced optimization with equal objective weights"
            },
            {
                "name": "emergency",
                "description": "Emergency mode prioritizing critical objectives (e.g., latency)"
            },
            {
                "name": "revenue_focused",
                "description": "Revenue-focused mode for recommendation systems"
            },
            {
                "name": "user_experience",
                "description": "User experience focused mode"
            }
        ]
    }


async def run_optimization_task(optimization_id: str, request: OptimizationRequest):
    """Background task to run optimization."""
    try:
        logger.info(f"Running optimization {optimization_id}")
        
        # Store the task in the engine
        task = asyncio.create_task(optimization_engine.optimize(request))
        optimization_engine._running_optimizations[optimization_id] = task
        
        # Wait for completion
        result = await task
        
        logger.info(f"Optimization {optimization_id} completed successfully")
        
        # Store result (in production, this would go to a database)
        optimization_engine._running_optimizations[optimization_id] = asyncio.create_task(
            asyncio.coroutine(lambda: result)()
        )
        
    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {e}")
        
        # Store error
        async def error_coroutine():
            raise e
        
        optimization_engine._running_optimizations[optimization_id] = asyncio.create_task(
            error_coroutine()
        )
