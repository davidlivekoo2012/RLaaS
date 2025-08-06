"""
RLaaS SDK Client - Main client interface.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import httpx
from dataclasses import dataclass
import time

from .models import (
    OptimizationRequest, OptimizationResult,
    TrainingRequest, TrainingResult,
    ModelInfo, DatasetInfo
)
from .utils import validate_config, format_response

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Client configuration."""
    api_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    verify_ssl: bool = True


class OptimizationClient:
    """Client for optimization operations."""
    
    def __init__(self, base_client: 'RLaaSClient'):
        self.base_client = base_client
    
    def start(
        self,
        problem_type: str,
        algorithm: str = "nsga3",
        mode: str = "normal",
        population_size: int = 100,
        generations: int = 500,
        weights: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> 'OptimizationJob':
        """
        Start a new optimization job.
        
        Args:
            problem_type: Type of problem ("5g" or "recommendation")
            algorithm: Optimization algorithm ("nsga3" or "moead")
            mode: Optimization mode ("normal", "emergency", etc.)
            population_size: Population size for the algorithm
            generations: Number of generations
            weights: Custom objective weights
            constraints: Problem constraints
            timeout: Timeout in seconds
            **kwargs: Additional parameters
            
        Returns:
            OptimizationJob instance
        """
        request = OptimizationRequest(
            problem_type=problem_type,
            algorithm=algorithm,
            mode=mode,
            population_size=population_size,
            generations=generations,
            weights=weights,
            constraints=constraints,
            timeout=timeout,
            **kwargs
        )
        
        # Start optimization
        response = self.base_client._post("/api/v1/optimization/optimize", request.to_dict())
        
        if "error" in response:
            raise RuntimeError(f"Failed to start optimization: {response['error']}")
        
        return OptimizationJob(
            optimization_id=response["optimization_id"],
            client=self.base_client
        )
    
    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List optimization jobs."""
        
        params = {}
        if status:
            params["status"] = status
        
        return self.base_client._get("/api/v1/optimization/jobs", params=params)
    
    def get(self, optimization_id: str) -> Dict[str, Any]:
        """Get optimization job details."""
        
        return self.base_client._get(f"/api/v1/optimization/optimize/{optimization_id}")
    
    def cancel(self, optimization_id: str) -> bool:
        """Cancel optimization job."""
        
        response = self.base_client._delete(f"/api/v1/optimization/optimize/{optimization_id}")
        return response.get("success", False)


class TrainingClient:
    """Client for training operations."""
    
    def __init__(self, base_client: 'RLaaSClient'):
        self.base_client = base_client
    
    def start(
        self,
        training_type: str,
        algorithm: str,
        dataset: str,
        hyperparameters: Dict[str, Any],
        resources: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'TrainingJob':
        """
        Start a new training job.
        
        Args:
            training_type: Type of training ("reinforcement_learning", "deep_learning", etc.)
            algorithm: Training algorithm
            dataset: Dataset identifier
            hyperparameters: Training hyperparameters
            resources: Resource requirements
            **kwargs: Additional parameters
            
        Returns:
            TrainingJob instance
        """
        request = TrainingRequest(
            training_type=training_type,
            algorithm=algorithm,
            dataset=dataset,
            hyperparameters=hyperparameters,
            resources=resources or {},
            **kwargs
        )
        
        response = self.base_client._post("/api/v1/training/jobs", request.to_dict())
        
        if "error" in response:
            raise RuntimeError(f"Failed to start training: {response['error']}")
        
        return TrainingJob(
            job_id=response["job_id"],
            client=self.base_client
        )
    
    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List training jobs."""
        
        params = {}
        if status:
            params["status"] = status
        
        return self.base_client._get("/api/v1/training/jobs", params=params)
    
    def get(self, job_id: str) -> Dict[str, Any]:
        """Get training job details."""
        
        return self.base_client._get(f"/api/v1/training/jobs/{job_id}")


class ModelClient:
    """Client for model operations."""
    
    def __init__(self, base_client: 'RLaaSClient'):
        self.base_client = base_client
    
    def list(self) -> List[ModelInfo]:
        """List deployed models."""
        
        response = self.base_client._get("/api/v1/inference/models")
        return [ModelInfo.from_dict(model) for model in response]
    
    def get(self, model_id: str) -> ModelInfo:
        """Get model information."""
        
        response = self.base_client._get(f"/api/v1/inference/models/{model_id}")
        return ModelInfo.from_dict(response)
    
    def deploy(
        self,
        model_id: str,
        version: str,
        resources: Optional[Dict[str, Any]] = None,
        scaling: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deploy a model for inference."""
        
        request = {
            "model_id": model_id,
            "version": version,
            "resources": resources or {},
            "scaling": scaling or {}
        }
        
        response = self.base_client._post("/api/v1/inference/deploy", request)
        return response["deployment_id"]
    
    def predict(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make prediction using deployed model."""
        
        request = {
            "model_id": model_id,
            "inputs": inputs,
            "options": options or {}
        }
        
        return self.base_client._post("/api/v1/inference/predict", request)
    
    def undeploy(self, model_id: str) -> bool:
        """Undeploy a model."""
        
        response = self.base_client._delete(f"/api/v1/inference/models/{model_id}")
        return response.get("success", False)


class DataClient:
    """Client for data operations."""
    
    def __init__(self, base_client: 'RLaaSClient'):
        self.base_client = base_client
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List available datasets."""
        
        response = self.base_client._get("/api/v1/data/datasets")
        return [DatasetInfo.from_dict(dataset) for dataset in response]
    
    def get_dataset(self, dataset_id: str) -> DatasetInfo:
        """Get dataset information."""
        
        response = self.base_client._get(f"/api/v1/data/datasets/{dataset_id}")
        return DatasetInfo.from_dict(response)
    
    def upload_dataset(
        self,
        file_path: str,
        name: str,
        description: Optional[str] = None
    ) -> str:
        """Upload a new dataset."""
        
        # This would implement file upload logic
        # For now, return a placeholder
        return "dataset_123"
    
    def list_features(self) -> List[Dict[str, Any]]:
        """List available features."""
        
        return self.base_client._get("/api/v1/data/features")
    
    def get_feature_values(
        self,
        feature_id: str,
        entity_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get feature values."""
        
        params = {}
        if entity_ids:
            params["entity_ids"] = ",".join(entity_ids)
        
        return self.base_client._get(f"/api/v1/data/features/{feature_id}", params=params)


class RLaaSClient:
    """
    Main RLaaS SDK client.
    
    Provides high-level interfaces for all RLaaS platform operations.
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True
    ):
        """
        Initialize RLaaS client.
        
        Args:
            api_url: RLaaS API base URL
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            verify_ssl: Whether to verify SSL certificates
        """
        self.config = ClientConfig(
            api_url=api_url.rstrip('/'),
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl
        )
        
        # Initialize sub-clients
        self.optimization = OptimizationClient(self)
        self.training = TrainingClient(self)
        self.models = ModelClient(self)
        self.data = DataClient(self)
        
        logger.info(f"RLaaS client initialized for {api_url}")
    
    def health(self) -> Dict[str, Any]:
        """Check platform health."""
        return self._get("/health")
    
    def version(self) -> str:
        """Get platform version."""
        health = self.health()
        return health.get("version", "unknown")
    
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request."""
        return self._request("GET", endpoint, params=params)
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request."""
        return self._request("POST", endpoint, json=data)
    
    def _put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make PUT request."""
        return self._request("PUT", endpoint, json=data)
    
    def _delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request."""
        return self._request("DELETE", endpoint)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retries."""
        
        url = f"{self.config.api_url}{endpoint}"
        headers = {}
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        for attempt in range(self.config.max_retries):
            try:
                with httpx.Client(
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl
                ) as client:
                    response = client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        json=json
                    )
                    
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.HTTPError as e:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"Request failed after {self.config.max_retries} attempts: {e}")
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        raise RuntimeError("Request failed")


class OptimizationJob:
    """Represents a running optimization job."""
    
    def __init__(self, optimization_id: str, client: RLaaSClient):
        self.optimization_id = optimization_id
        self.client = client
    
    def status(self) -> Dict[str, Any]:
        """Get current status."""
        return self.client.optimization.get(self.optimization_id)
    
    def wait_for_completion(self, poll_interval: int = 5) -> OptimizationResult:
        """Wait for optimization to complete."""
        
        while True:
            status = self.status()
            
            if status["status"] == "completed":
                return OptimizationResult.from_dict(status)
            elif status["status"] == "failed":
                raise RuntimeError(f"Optimization failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "cancelled":
                raise RuntimeError("Optimization was cancelled")
            
            time.sleep(poll_interval)
    
    def cancel(self) -> bool:
        """Cancel the optimization."""
        return self.client.optimization.cancel(self.optimization_id)


class TrainingJob:
    """Represents a running training job."""
    
    def __init__(self, job_id: str, client: RLaaSClient):
        self.job_id = job_id
        self.client = client
    
    def status(self) -> Dict[str, Any]:
        """Get current status."""
        return self.client.training.get(self.job_id)
    
    def wait_for_completion(self, poll_interval: int = 10) -> TrainingResult:
        """Wait for training to complete."""
        
        while True:
            status = self.status()
            
            if status["status"] == "completed":
                return TrainingResult.from_dict(status)
            elif status["status"] == "failed":
                raise RuntimeError(f"Training failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "cancelled":
                raise RuntimeError("Training was cancelled")
            
            time.sleep(poll_interval)
    
    def cancel(self) -> bool:
        """Cancel the training."""
        # Implementation would call training cancel endpoint
        return True
