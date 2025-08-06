"""
Experiment Management for RLaaS platform.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, Run
import wandb

from rlaas.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class ExperimentBackend(Enum):
    """Experiment tracking backends."""
    MLFLOW = "mlflow"
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"


class ExperimentStatus(Enum):
    """Experiment status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    backend: ExperimentBackend = ExperimentBackend.MLFLOW
    experiment_name: str = "rlaas_experiment"
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Backend-specific configs
    mlflow_config: Dict[str, Any] = field(default_factory=dict)
    wandb_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """Experiment run information."""
    run_id: str
    experiment_id: str
    name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Metrics and parameters
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


class ExperimentManager:
    """
    Experiment tracking and management.
    
    Supports multiple experiment tracking backends:
    - MLflow
    - Weights & Biases (wandb)
    - TensorBoard
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = None
        self.current_run = None
        self.experiments: Dict[str, str] = {}  # name -> id mapping
        
        self._initialize_backend()
        
        logger.info(f"ExperimentManager initialized with {config.backend.value}")
    
    def _initialize_backend(self):
        """Initialize experiment tracking backend."""
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            self._initialize_mlflow()
        elif self.config.backend == ExperimentBackend.WANDB:
            self._initialize_wandb()
        elif self.config.backend == ExperimentBackend.TENSORBOARD:
            self._initialize_tensorboard()
    
    def _initialize_mlflow(self):
        """Initialize MLflow."""
        
        # Set tracking URI
        tracking_uri = self.config.mlflow_config.get(
            "tracking_uri", 
            config.mlflow.tracking_uri
        )
        mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize client
        self.client = MlflowClient(tracking_uri)
        
        logger.info(f"MLflow initialized with tracking URI: {tracking_uri}")
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases."""
        
        # Set up wandb config
        wandb_config = {
            "project": self.config.wandb_config.get("project", "rlaas"),
            "entity": self.config.wandb_config.get("entity"),
            "mode": self.config.wandb_config.get("mode", "online")
        }
        
        # Initialize wandb
        wandb.init(**wandb_config)
        
        logger.info("Weights & Biases initialized")
    
    def _initialize_tensorboard(self):
        """Initialize TensorBoard."""
        
        # TensorBoard initialization would go here
        # For now, just log that it's prepared
        logger.info("TensorBoard backend prepared")
    
    async def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Experiment tags
            
        Returns:
            Experiment ID
        """
        tags = tags or {}
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            return await self._create_mlflow_experiment(name, description, tags)
        elif self.config.backend == ExperimentBackend.WANDB:
            return await self._create_wandb_experiment(name, description, tags)
        else:
            # For other backends, generate a simple ID
            experiment_id = f"{name}_{int(datetime.now().timestamp())}"
            self.experiments[name] = experiment_id
            return experiment_id
    
    async def _create_mlflow_experiment(
        self,
        name: str,
        description: str,
        tags: Dict[str, str]
    ) -> str:
        """Create MLflow experiment."""
        
        try:
            # Check if experiment exists
            experiment = self.client.get_experiment_by_name(name)
            if experiment:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {name} ({experiment_id})")
            else:
                # Create new experiment
                experiment_id = self.client.create_experiment(
                    name=name,
                    artifact_location=self.config.mlflow_config.get("artifact_location"),
                    tags=tags
                )
                logger.info(f"Created MLflow experiment: {name} ({experiment_id})")
            
            self.experiments[name] = experiment_id
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create MLflow experiment: {e}")
            raise
    
    async def _create_wandb_experiment(
        self,
        name: str,
        description: str,
        tags: Dict[str, str]
    ) -> str:
        """Create wandb experiment (project)."""
        
        # In wandb, experiments are typically projects
        # We'll use the run name as experiment identifier
        experiment_id = f"wandb_{name}_{int(datetime.now().timestamp())}"
        self.experiments[name] = experiment_id
        
        logger.info(f"Wandb experiment prepared: {name}")
        return experiment_id
    
    async def start_run(
        self,
        experiment_id: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ExperimentRun:
        """
        Start a new experiment run.
        
        Args:
            experiment_id: Experiment ID
            run_name: Run name
            tags: Run tags
            
        Returns:
            Experiment run
        """
        tags = tags or {}
        run_name = run_name or f"run_{int(datetime.now().timestamp())}"
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            return await self._start_mlflow_run(experiment_id, run_name, tags)
        elif self.config.backend == ExperimentBackend.WANDB:
            return await self._start_wandb_run(experiment_id, run_name, tags)
        else:
            # Simple run for other backends
            run = ExperimentRun(
                run_id=f"run_{int(datetime.now().timestamp())}",
                experiment_id=experiment_id,
                name=run_name,
                status=ExperimentStatus.RUNNING,
                start_time=datetime.now(),
                tags=tags
            )
            self.current_run = run
            return run
    
    async def _start_mlflow_run(
        self,
        experiment_id: str,
        run_name: str,
        tags: Dict[str, str]
    ) -> ExperimentRun:
        """Start MLflow run."""
        
        # Start MLflow run
        mlflow_run = self.client.create_run(
            experiment_id=experiment_id,
            tags=tags
        )
        
        # Set run name
        self.client.set_tag(mlflow_run.info.run_id, "mlflow.runName", run_name)
        
        # Create our run object
        run = ExperimentRun(
            run_id=mlflow_run.info.run_id,
            experiment_id=experiment_id,
            name=run_name,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.fromtimestamp(mlflow_run.info.start_time / 1000),
            tags=tags
        )
        
        self.current_run = run
        
        logger.info(f"MLflow run started: {run_name} ({run.run_id})")
        return run
    
    async def _start_wandb_run(
        self,
        experiment_id: str,
        run_name: str,
        tags: Dict[str, str]
    ) -> ExperimentRun:
        """Start wandb run."""
        
        # Start wandb run
        wandb_run = wandb.init(
            name=run_name,
            tags=list(tags.keys()),
            reinit=True
        )
        
        # Create our run object
        run = ExperimentRun(
            run_id=wandb_run.id,
            experiment_id=experiment_id,
            name=run_name,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now(),
            tags=tags
        )
        
        self.current_run = run
        
        logger.info(f"Wandb run started: {run_name} ({run.run_id})")
        return run
    
    async def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        
        if not self.current_run:
            raise RuntimeError("No active run")
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            for key, value in params.items():
                self.client.log_param(self.current_run.run_id, key, value)
        
        elif self.config.backend == ExperimentBackend.WANDB:
            wandb.config.update(params)
        
        # Update our run object
        self.current_run.params.update(params)
        
        logger.debug(f"Logged parameters: {params}")
    
    async def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        
        if not self.current_run:
            raise RuntimeError("No active run")
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            for key, value in metrics.items():
                self.client.log_metric(self.current_run.run_id, key, value, step=step)
        
        elif self.config.backend == ExperimentBackend.WANDB:
            log_data = metrics.copy()
            if step is not None:
                log_data["step"] = step
            wandb.log(log_data)
        
        # Update our run object
        self.current_run.metrics.update(metrics)
        
        logger.debug(f"Logged metrics: {metrics}")
    
    async def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact."""
        
        if not self.current_run:
            raise RuntimeError("No active run")
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            self.client.log_artifact(self.current_run.run_id, artifact_path)
        
        elif self.config.backend == ExperimentBackend.WANDB:
            wandb.save(artifact_path, base_path=artifact_name)
        
        # Update our run object
        self.current_run.artifacts.append(artifact_path)
        
        logger.debug(f"Logged artifact: {artifact_path}")
    
    async def log_model(
        self,
        model,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log model."""
        
        if not self.current_run:
            raise RuntimeError("No active run")
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            # Log model with MLflow
            import mlflow.pytorch
            import mlflow.sklearn
            
            # Determine model type and log accordingly
            if hasattr(model, 'state_dict'):  # PyTorch model
                mlflow.pytorch.log_model(model, model_name)
            elif hasattr(model, 'predict'):  # Sklearn-like model
                mlflow.sklearn.log_model(model, model_name)
            else:
                # Generic model logging
                mlflow.log_artifact(model, model_name)
        
        elif self.config.backend == ExperimentBackend.WANDB:
            # Save model with wandb
            model_artifact = wandb.Artifact(model_name, type="model")
            model_artifact.add_file(model)
            wandb.log_artifact(model_artifact)
        
        logger.info(f"Logged model: {model_name}")
    
    async def end_run(self, status: ExperimentStatus = ExperimentStatus.COMPLETED):
        """End current run."""
        
        if not self.current_run:
            return
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            mlflow_status = "FINISHED" if status == ExperimentStatus.COMPLETED else "FAILED"
            self.client.set_terminated(self.current_run.run_id, mlflow_status)
        
        elif self.config.backend == ExperimentBackend.WANDB:
            wandb.finish()
        
        # Update our run object
        self.current_run.status = status
        self.current_run.end_time = datetime.now()
        
        logger.info(f"Run ended: {self.current_run.run_id} ({status.value})")
        
        self.current_run = None
    
    async def get_experiment_runs(self, experiment_id: str) -> List[ExperimentRun]:
        """Get all runs for an experiment."""
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["start_time DESC"]
            )
            
            experiment_runs = []
            for run in runs:
                experiment_run = ExperimentRun(
                    run_id=run.info.run_id,
                    experiment_id=experiment_id,
                    name=run.data.tags.get("mlflow.runName", "Unnamed"),
                    status=ExperimentStatus(run.info.status.lower()),
                    start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                    end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    params=run.data.params,
                    metrics=run.data.metrics,
                    tags=run.data.tags
                )
                experiment_runs.append(experiment_run)
            
            return experiment_runs
        
        else:
            # For other backends, return empty list for now
            return []
    
    async def get_run_details(self, run_id: str) -> Optional[ExperimentRun]:
        """Get detailed information about a run."""
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            try:
                run = self.client.get_run(run_id)
                
                return ExperimentRun(
                    run_id=run.info.run_id,
                    experiment_id=run.info.experiment_id,
                    name=run.data.tags.get("mlflow.runName", "Unnamed"),
                    status=ExperimentStatus(run.info.status.lower()),
                    start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                    end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    params=run.data.params,
                    metrics=run.data.metrics,
                    tags=run.data.tags
                )
            except Exception as e:
                logger.error(f"Failed to get run details: {e}")
                return None
        
        return None
    
    async def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        
        runs = []
        for run_id in run_ids:
            run_details = await self.get_run_details(run_id)
            if run_details:
                runs.append(run_details)
        
        if not runs:
            return {}
        
        # Extract common metrics for comparison
        all_metrics = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())
        
        comparison = {
            "runs": [
                {
                    "run_id": run.run_id,
                    "name": run.name,
                    "metrics": {metric: run.metrics.get(metric) for metric in all_metrics}
                }
                for run in runs
            ],
            "metrics": list(all_metrics)
        }
        
        return comparison
    
    async def search_runs(
        self,
        experiment_id: str,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None
    ) -> List[ExperimentRun]:
        """Search runs with filters."""
        
        if self.config.backend == ExperimentBackend.MLFLOW:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                order_by=order_by or ["start_time DESC"]
            )
            
            experiment_runs = []
            for run in runs:
                experiment_run = ExperimentRun(
                    run_id=run.info.run_id,
                    experiment_id=experiment_id,
                    name=run.data.tags.get("mlflow.runName", "Unnamed"),
                    status=ExperimentStatus(run.info.status.lower()),
                    start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                    end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    params=run.data.params,
                    metrics=run.data.metrics,
                    tags=run.data.tags
                )
                experiment_runs.append(experiment_run)
            
            return experiment_runs
        
        else:
            return []
