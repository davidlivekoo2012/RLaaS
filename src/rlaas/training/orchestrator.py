"""
Training Orchestrator for managing ML training workflows.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import json
import mlflow
from mlflow.tracking import MlflowClient

from rlaas.config import get_config
from rlaas.core.scheduler import AdaptiveScheduler, Task, TaskType, TaskPriority

logger = logging.getLogger(__name__)
config = get_config()


class TrainingStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingType(Enum):
    """Types of training jobs."""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DEEP_LEARNING = "deep_learning"
    OPTIMIZATION = "optimization"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"


@dataclass
class TrainingJob:
    """Training job definition."""
    job_id: str
    name: str
    training_type: TrainingType
    algorithm: str
    dataset: str
    hyperparameters: Dict[str, Any]
    resources: Dict[str, Any]
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Runtime information
    status: TrainingStatus = TrainingStatus.PENDING
    mlflow_run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


@dataclass
class TrainingPipeline:
    """Training pipeline definition."""
    pipeline_id: str
    name: str
    description: str
    jobs: List[TrainingJob]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression
    
    # Runtime information
    status: TrainingStatus = TrainingStatus.PENDING
    current_job: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class TrainingOrchestrator:
    """
    Training Orchestrator for managing ML training workflows.
    
    Provides capabilities for:
    - Training job scheduling and execution
    - Pipeline orchestration
    - Resource management
    - Experiment tracking integration
    """
    
    def __init__(self):
        # Initialize MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Initialize scheduler
        self.scheduler = AdaptiveScheduler()
        
        # Job management
        self.jobs: Dict[str, TrainingJob] = {}
        self.pipelines: Dict[str, TrainingPipeline] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        
        logger.info("TrainingOrchestrator initialized")
    
    async def start(self):
        """Start the training orchestrator."""
        await self.scheduler.start()
        logger.info("Training orchestrator started")
    
    async def stop(self):
        """Stop the training orchestrator."""
        # Cancel all running jobs
        for job_id, task in self.running_jobs.items():
            task.cancel()
            logger.info(f"Cancelled training job: {job_id}")
        
        await self.scheduler.stop()
        logger.info("Training orchestrator stopped")
    
    async def submit_job(self, job: TrainingJob) -> str:
        """
        Submit a training job for execution.
        
        Args:
            job: Training job definition
            
        Returns:
            Job ID
        """
        # Store job
        self.jobs[job.job_id] = job
        
        # Create scheduler task
        scheduler_task = Task(
            task_id=job.job_id,
            task_type=TaskType.TRAINING,
            priority=TaskPriority.NORMAL,
            resource_requirements=job.resources,
            estimated_duration=self._estimate_training_duration(job),
            metadata={
                "training_type": job.training_type.value,
                "algorithm": job.algorithm,
                "dataset": job.dataset
            }
        )
        
        # Submit to scheduler
        await self.scheduler.submit_task(scheduler_task)
        
        # Start job execution
        execution_task = asyncio.create_task(self._execute_job(job))
        self.running_jobs[job.job_id] = execution_task
        
        logger.info(f"Training job submitted: {job.job_id}")
        return job.job_id
    
    async def submit_pipeline(self, pipeline: TrainingPipeline) -> str:
        """
        Submit a training pipeline for execution.
        
        Args:
            pipeline: Training pipeline definition
            
        Returns:
            Pipeline ID
        """
        # Store pipeline
        self.pipelines[pipeline.pipeline_id] = pipeline
        
        # Submit individual jobs with dependencies
        for job in pipeline.jobs:
            # Set dependencies
            dependencies = pipeline.dependencies.get(job.job_id, [])
            
            scheduler_task = Task(
                task_id=job.job_id,
                task_type=TaskType.TRAINING,
                priority=TaskPriority.NORMAL,
                resource_requirements=job.resources,
                estimated_duration=self._estimate_training_duration(job),
                dependencies=dependencies,
                metadata={
                    "pipeline_id": pipeline.pipeline_id,
                    "training_type": job.training_type.value,
                    "algorithm": job.algorithm
                }
            )
            
            await self.scheduler.submit_task(scheduler_task)
            self.jobs[job.job_id] = job
        
        # Start pipeline execution
        execution_task = asyncio.create_task(self._execute_pipeline(pipeline))
        self.running_jobs[pipeline.pipeline_id] = execution_task
        
        logger.info(f"Training pipeline submitted: {pipeline.pipeline_id}")
        return pipeline.pipeline_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status."""
        
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        # Get scheduler status
        scheduler_status = await self.scheduler.get_task_status(job_id)
        
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "training_type": job.training_type.value,
            "algorithm": job.algorithm,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "mlflow_run_id": job.mlflow_run_id,
            "metrics": job.metrics,
            "scheduler_status": scheduler_status,
            "logs": job.logs[-10:]  # Last 10 log entries
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        
        if job_id not in self.jobs:
            return False
        
        # Cancel scheduler task
        await self.scheduler.cancel_task(job_id)
        
        # Cancel execution task
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            del self.running_jobs[job_id]
        
        # Update job status
        job = self.jobs[job_id]
        job.status = TrainingStatus.CANCELLED
        job.end_time = datetime.now()
        
        logger.info(f"Training job cancelled: {job_id}")
        return True
    
    async def _execute_job(self, job: TrainingJob):
        """Execute a training job."""
        
        try:
            # Update job status
            job.status = TrainingStatus.RUNNING
            job.start_time = datetime.now()
            
            # Create MLflow experiment
            experiment_name = f"rlaas_{job.training_type.value}_{job.algorithm}"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    job.experiment_id = experiment.experiment_id
                else:
                    job.experiment_id = mlflow.create_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"Failed to create MLflow experiment: {e}")
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=job.experiment_id) as run:
                job.mlflow_run_id = run.info.run_id
                
                # Log hyperparameters
                for key, value in job.hyperparameters.items():
                    mlflow.log_param(key, value)
                
                # Log job metadata
                mlflow.log_param("training_type", job.training_type.value)
                mlflow.log_param("algorithm", job.algorithm)
                mlflow.log_param("dataset", job.dataset)
                
                # Execute training based on type
                if job.training_type == TrainingType.REINFORCEMENT_LEARNING:
                    await self._execute_rl_training(job)
                elif job.training_type == TrainingType.DEEP_LEARNING:
                    await self._execute_dl_training(job)
                elif job.training_type == TrainingType.OPTIMIZATION:
                    await self._execute_optimization_training(job)
                else:
                    raise ValueError(f"Unsupported training type: {job.training_type}")
                
                # Log final metrics
                for key, value in job.metrics.items():
                    mlflow.log_metric(key, value)
            
            # Update job status
            job.status = TrainingStatus.COMPLETED
            job.end_time = datetime.now()
            
            logger.info(f"Training job completed: {job.job_id}")
            
        except asyncio.CancelledError:
            job.status = TrainingStatus.CANCELLED
            job.end_time = datetime.now()
            logger.info(f"Training job cancelled: {job.job_id}")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.end_time = datetime.now()
            job.logs.append(f"Error: {str(e)}")
            logger.error(f"Training job failed: {job.job_id} - {e}")
        
        finally:
            # Clean up
            self.running_jobs.pop(job.job_id, None)
    
    async def _execute_rl_training(self, job: TrainingJob):
        """Execute reinforcement learning training."""
        
        from rlaas.core.policy import PolicyEngine, PolicyConfig, PolicyType, EnvironmentType
        
        # Create policy engine
        policy_engine = PolicyEngine()
        
        # Parse configuration
        policy_type = PolicyType(job.algorithm.lower())
        env_type = EnvironmentType(job.hyperparameters.get("environment", "network_5g"))
        
        config = PolicyConfig(
            policy_type=policy_type,
            environment_type=env_type,
            total_timesteps=job.hyperparameters.get("total_timesteps", 100000),
            learning_rate=job.hyperparameters.get("learning_rate", 3e-4),
            batch_size=job.hyperparameters.get("batch_size", 256),
        )
        
        # Create and train policy
        policy_id = await policy_engine.create_policy(job.job_id, config)
        result = await policy_engine.train_policy(policy_id, config, background=False)
        
        # Store results
        job.metrics.update({
            "final_reward": result.final_reward,
            "training_time": result.training_time,
            "convergence_achieved": result.convergence_achieved,
            "total_episodes": len(result.rewards)
        })
        
        # Save model
        model_path = f"/tmp/models/{job.job_id}"
        await policy_engine.save_policy(policy_id, model_path)
        job.artifacts["model_path"] = model_path
    
    async def _execute_dl_training(self, job: TrainingJob):
        """Execute deep learning training."""
        # Placeholder for deep learning training
        # This would integrate with frameworks like PyTorch, TensorFlow
        
        import time
        import numpy as np
        
        # Simulate training
        epochs = job.hyperparameters.get("epochs", 10)
        
        for epoch in range(epochs):
            # Simulate epoch training
            await asyncio.sleep(1)  # Simulate training time
            
            # Simulate metrics
            loss = np.random.exponential(0.5) * np.exp(-epoch * 0.1)
            accuracy = 1.0 - np.exp(-epoch * 0.2) + np.random.normal(0, 0.01)
            
            job.metrics.update({
                f"epoch_{epoch}_loss": loss,
                f"epoch_{epoch}_accuracy": accuracy
            })
            
            job.logs.append(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        # Final metrics
        job.metrics.update({
            "final_loss": loss,
            "final_accuracy": accuracy,
            "epochs_completed": epochs
        })
    
    async def _execute_optimization_training(self, job: TrainingJob):
        """Execute optimization training."""
        
        from rlaas.core.optimization import OptimizationEngine, OptimizationRequest
        from rlaas.core.optimization import OptimizationAlgorithm, OptimizationMode
        from rlaas.core.optimization import create_5g_optimization_problem
        
        # Create optimization engine
        opt_engine = OptimizationEngine()
        
        # Parse configuration
        algorithm = OptimizationAlgorithm(job.algorithm.lower())
        mode = OptimizationMode(job.hyperparameters.get("mode", "normal"))
        
        # Create problem
        if job.hyperparameters.get("problem_type") == "5g":
            problem = create_5g_optimization_problem()
        else:
            problem = create_5g_optimization_problem()  # Default
        
        # Create optimization request
        request = OptimizationRequest(
            problem=problem,
            algorithm=algorithm,
            mode=mode,
            population_size=job.hyperparameters.get("population_size", 100),
            generations=job.hyperparameters.get("generations", 500)
        )
        
        # Run optimization
        result = await opt_engine.optimize(request)
        
        # Store results
        job.metrics.update({
            "execution_time": result.execution_time,
            "n_solutions": len(result.pareto_frontier.solutions),
            "best_solution_id": result.best_solution.id
        })
        
        # Store best solution objectives
        for obj_name, obj_value in result.best_solution.objectives.items():
            job.metrics[f"best_{obj_name}"] = obj_value
    
    async def _execute_pipeline(self, pipeline: TrainingPipeline):
        """Execute a training pipeline."""
        
        try:
            pipeline.status = TrainingStatus.RUNNING
            pipeline.start_time = datetime.now()
            
            # Execute jobs in dependency order
            completed_jobs = set()
            
            while len(completed_jobs) < len(pipeline.jobs):
                # Find jobs ready to execute
                ready_jobs = []
                
                for job in pipeline.jobs:
                    if job.job_id in completed_jobs:
                        continue
                    
                    # Check dependencies
                    dependencies = pipeline.dependencies.get(job.job_id, [])
                    if all(dep in completed_jobs for dep in dependencies):
                        ready_jobs.append(job)
                
                if not ready_jobs:
                    raise RuntimeError("Pipeline has circular dependencies or unresolvable dependencies")
                
                # Execute ready jobs
                tasks = []
                for job in ready_jobs:
                    pipeline.current_job = job.job_id
                    task = asyncio.create_task(self._execute_job(job))
                    tasks.append((job.job_id, task))
                
                # Wait for jobs to complete
                for job_id, task in tasks:
                    await task
                    completed_jobs.add(job_id)
            
            pipeline.status = TrainingStatus.COMPLETED
            pipeline.end_time = datetime.now()
            
            logger.info(f"Training pipeline completed: {pipeline.pipeline_id}")
            
        except Exception as e:
            pipeline.status = TrainingStatus.FAILED
            pipeline.end_time = datetime.now()
            logger.error(f"Training pipeline failed: {pipeline.pipeline_id} - {e}")
        
        finally:
            self.running_jobs.pop(pipeline.pipeline_id, None)
    
    def _estimate_training_duration(self, job: TrainingJob) -> float:
        """Estimate training duration in seconds."""
        
        if job.training_type == TrainingType.REINFORCEMENT_LEARNING:
            timesteps = job.hyperparameters.get("total_timesteps", 100000)
            return timesteps / 1000  # Rough estimate
        
        elif job.training_type == TrainingType.DEEP_LEARNING:
            epochs = job.hyperparameters.get("epochs", 10)
            return epochs * 60  # Rough estimate: 1 minute per epoch
        
        elif job.training_type == TrainingType.OPTIMIZATION:
            generations = job.hyperparameters.get("generations", 500)
            return generations * 2  # Rough estimate: 2 seconds per generation
        
        else:
            return 3600  # Default: 1 hour
    
    async def list_jobs(self, status: Optional[TrainingStatus] = None) -> List[Dict[str, Any]]:
        """List training jobs."""
        
        jobs = []
        for job in self.jobs.values():
            if status is None or job.status == status:
                job_info = await self.get_job_status(job.job_id)
                if job_info:
                    jobs.append(job_info)
        
        return jobs
    
    async def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        
        total_jobs = len(self.jobs)
        running_jobs = len([j for j in self.jobs.values() if j.status == TrainingStatus.RUNNING])
        completed_jobs = len([j for j in self.jobs.values() if j.status == TrainingStatus.COMPLETED])
        failed_jobs = len([j for j in self.jobs.values() if j.status == TrainingStatus.FAILED])
        
        return {
            "total_jobs": total_jobs,
            "running_jobs": running_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
            "total_pipelines": len(self.pipelines),
            "scheduler_stats": await self.scheduler.get_performance_metrics()
        }
