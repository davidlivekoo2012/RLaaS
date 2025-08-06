"""
RLaaS Python SDK

A comprehensive Python SDK for the RLaaS (Reinforcement Learning as a Service) platform.

This SDK provides high-level interfaces for:
- Multi-objective optimization
- Reinforcement learning training
- Model management and deployment
- Data management
- Monitoring and analytics

Example usage:
    ```python
    from rlaas.sdk import RLaaSClient
    
    # Initialize client
    client = RLaaSClient(api_url="https://api.rlaas.ai")
    
    # Start optimization
    optimization = client.optimization.start(
        problem_type="5g",
        algorithm="nsga3",
        population_size=100,
        generations=500
    )
    
    # Monitor progress
    result = optimization.wait_for_completion()
    print(f"Best solution: {result.best_solution}")
    ```
"""

from .client import RLaaSClient
from .models import (
    OptimizationRequest,
    OptimizationResult,
    TrainingRequest,
    TrainingResult,
    ModelInfo,
    DatasetInfo
)

__version__ = "0.1.0"
__author__ = "RLaaS Team"
__email__ = "team@rlaas.ai"

__all__ = [
    "RLaaSClient",
    "OptimizationRequest",
    "OptimizationResult", 
    "TrainingRequest",
    "TrainingResult",
    "ModelInfo",
    "DatasetInfo",
]
