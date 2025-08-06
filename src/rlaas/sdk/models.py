"""
Data models for RLaaS SDK.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class OptimizationAlgorithm(Enum):
    """Optimization algorithms."""
    NSGA3 = "nsga3"
    MOEAD = "moead"
    SPEA2 = "spea2"


class OptimizationMode(Enum):
    """Optimization modes."""
    NORMAL = "normal"
    EMERGENCY = "emergency"
    REVENUE_FOCUSED = "revenue_focused"
    USER_EXPERIENCE = "user_experience"


class OptimizationStatus(Enum):
    """Optimization job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingType(Enum):
    """Training job types."""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DEEP_LEARNING = "deep_learning"
    OPTIMIZATION = "optimization"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"


class TrainingStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelStatus(Enum):
    """Model deployment status."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"


@dataclass
class OptimizationRequest:
    """Optimization request model."""
    problem_type: str
    algorithm: str = "nsga3"
    mode: str = "normal"
    population_size: int = 100
    generations: int = 500
    weights: Optional[Dict[str, float]] = None
    constraints: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    
    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "problem_type": self.problem_type,
            "algorithm": self.algorithm,
            "mode": self.mode,
            "population_size": self.population_size,
            "generations": self.generations,
            **self.parameters
        }
        
        if self.weights:
            data["weights"] = self.weights
        if self.constraints:
            data["constraints"] = self.constraints
        if self.timeout:
            data["timeout"] = self.timeout
        
        return data


@dataclass
class Solution:
    """Optimization solution."""
    id: str
    variables: Dict[str, float]
    objectives: Dict[str, float]
    constraints: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Solution':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            variables=data["variables"],
            objectives=data["objectives"],
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class ParetoFrontier:
    """Pareto frontier of solutions."""
    solutions: List[Solution]
    objectives: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParetoFrontier':
        """Create from dictionary."""
        solutions = [Solution.from_dict(sol) for sol in data["solutions"]]
        return cls(
            solutions=solutions,
            objectives=data["objectives"]
        )


@dataclass
class OptimizationResult:
    """Optimization result."""
    optimization_id: str
    status: OptimizationStatus
    best_solution: Optional[Solution] = None
    pareto_frontier: Optional[ParetoFrontier] = None
    
    # Execution info
    execution_time: float = 0.0
    generations_completed: int = 0
    convergence_achieved: bool = False
    
    # Metadata
    algorithm: str = ""
    problem_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """Create from dictionary."""
        
        best_solution = None
        if "best_solution" in data and data["best_solution"]:
            best_solution = Solution.from_dict(data["best_solution"])
        
        pareto_frontier = None
        if "pareto_frontier" in data and data["pareto_frontier"]:
            pareto_frontier = ParetoFrontier.from_dict(data["pareto_frontier"])
        
        return cls(
            optimization_id=data["optimization_id"],
            status=OptimizationStatus(data["status"]),
            best_solution=best_solution,
            pareto_frontier=pareto_frontier,
            execution_time=data.get("execution_time", 0.0),
            generations_completed=data.get("generations_completed", 0),
            convergence_achieved=data.get("convergence_achieved", False),
            algorithm=data.get("algorithm", ""),
            problem_type=data.get("problem_type", ""),
            parameters=data.get("parameters", {}),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        )


@dataclass
class TrainingRequest:
    """Training request model."""
    training_type: str
    algorithm: str
    dataset: str
    hyperparameters: Dict[str, Any]
    resources: Dict[str, Any] = field(default_factory=dict)
    
    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "training_type": self.training_type,
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "hyperparameters": self.hyperparameters,
            "resources": self.resources,
            **self.parameters
        }


@dataclass
class TrainingMetrics:
    """Training metrics."""
    epoch: int
    loss: float
    accuracy: Optional[float] = None
    reward: Optional[float] = None
    
    # Additional metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """Create from dictionary."""
        return cls(
            epoch=data["epoch"],
            loss=data["loss"],
            accuracy=data.get("accuracy"),
            reward=data.get("reward"),
            metrics=data.get("metrics", {})
        )


@dataclass
class TrainingResult:
    """Training result."""
    job_id: str
    status: TrainingStatus
    
    # Training info
    final_metrics: Optional[TrainingMetrics] = None
    training_history: List[TrainingMetrics] = field(default_factory=list)
    
    # Model info
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    model_path: Optional[str] = None
    
    # Execution info
    training_time: float = 0.0
    epochs_completed: int = 0
    convergence_achieved: bool = False
    
    # Metadata
    training_type: str = ""
    algorithm: str = ""
    dataset: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingResult':
        """Create from dictionary."""
        
        final_metrics = None
        if "final_metrics" in data and data["final_metrics"]:
            final_metrics = TrainingMetrics.from_dict(data["final_metrics"])
        
        training_history = []
        if "training_history" in data:
            training_history = [
                TrainingMetrics.from_dict(m) for m in data["training_history"]
            ]
        
        return cls(
            job_id=data["job_id"],
            status=TrainingStatus(data["status"]),
            final_metrics=final_metrics,
            training_history=training_history,
            model_id=data.get("model_id"),
            model_version=data.get("model_version"),
            model_path=data.get("model_path"),
            training_time=data.get("training_time", 0.0),
            epochs_completed=data.get("epochs_completed", 0),
            convergence_achieved=data.get("convergence_achieved", False),
            training_type=data.get("training_type", ""),
            algorithm=data.get("algorithm", ""),
            dataset=data.get("dataset", ""),
            hyperparameters=data.get("hyperparameters", {}),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        )


@dataclass
class ModelInfo:
    """Model information."""
    model_id: str
    name: str
    version: str
    framework: str
    
    # Model details
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Deployment info
    status: ModelStatus = ModelStatus.LOADING
    endpoint_url: Optional[str] = None
    
    # Performance metrics
    request_count: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            name=data["name"],
            version=data["version"],
            framework=data["framework"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            status=ModelStatus(data.get("status", "loading")),
            endpoint_url=data.get("endpoint_url"),
            request_count=data.get("request_count", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            error_rate=data.get("error_rate", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )


@dataclass
class DatasetInfo:
    """Dataset information."""
    dataset_id: str
    name: str
    description: str
    
    # Dataset details
    format: str = "parquet"
    size_bytes: int = 0
    row_count: int = 0
    column_count: int = 0
    
    # Schema
    schema: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """Create from dictionary."""
        return cls(
            dataset_id=data["dataset_id"],
            name=data["name"],
            description=data["description"],
            format=data.get("format", "parquet"),
            size_bytes=data.get("size_bytes", 0),
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            schema=data.get("schema", {}),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )


@dataclass
class InferenceRequest:
    """Inference request model."""
    model_id: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "inputs": self.inputs,
            "parameters": self.parameters
        }


@dataclass
class InferenceResponse:
    """Inference response model."""
    model_id: str
    predictions: Dict[str, Any]
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    
    # Metadata
    model_version: str = ""
    timestamp: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceResponse':
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            predictions=data["predictions"],
            confidence=data.get("confidence"),
            latency_ms=data.get("latency_ms", 0.0),
            model_version=data.get("model_version", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        )


@dataclass
class ExperimentInfo:
    """Experiment information."""
    experiment_id: str
    name: str
    description: str
    
    # Experiment details
    status: str = "created"
    run_count: int = 0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentInfo':
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data["description"],
            status=data.get("status", "created"),
            run_count=data.get("run_count", 0),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )


@dataclass
class SystemHealth:
    """System health information."""
    status: str
    version: str
    uptime: float
    environment: str
    
    # Component health
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemHealth':
        """Create from dictionary."""
        return cls(
            status=data["status"],
            version=data["version"],
            uptime=data["uptime"],
            environment=data["environment"],
            components=data.get("components", {}),
            cpu_usage=data.get("cpu_usage", 0.0),
            memory_usage=data.get("memory_usage", 0.0),
            disk_usage=data.get("disk_usage", 0.0)
        )


# Error models
@dataclass
class APIError:
    """API error model."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIError':
        """Create from dictionary."""
        return cls(
            error_code=data["error_code"],
            message=data["message"],
            details=data.get("details")
        )


class RLaaSException(Exception):
    """Base exception for RLaaS SDK."""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class OptimizationError(RLaaSException):
    """Optimization-related error."""
    pass


class TrainingError(RLaaSException):
    """Training-related error."""
    pass


class ModelError(RLaaSException):
    """Model-related error."""
    pass


class DataError(RLaaSException):
    """Data-related error."""
    pass


class APIConnectionError(RLaaSException):
    """API connection error."""
    pass


class AuthenticationError(RLaaSException):
    """Authentication error."""
    pass


class ValidationError(RLaaSException):
    """Validation error."""
    pass
