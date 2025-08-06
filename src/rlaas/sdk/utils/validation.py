"""
Validation utilities for RLaaS SDK.
"""

from typing import Dict, List, Any, Optional
import re
from ..models import ValidationError


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate client configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If configuration is invalid
    """
    required_fields = ["api_url"]
    
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate API URL format
    api_url = config["api_url"]
    if not isinstance(api_url, str):
        raise ValidationError("api_url must be a string")
    
    if not api_url.startswith(("http://", "https://")):
        raise ValidationError("api_url must start with http:// or https://")
    
    # Validate timeout if provided
    if "timeout" in config:
        timeout = config["timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError("timeout must be a positive number")
    
    # Validate max_retries if provided
    if "max_retries" in config:
        max_retries = config["max_retries"]
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValidationError("max_retries must be a non-negative integer")
    
    return True


def validate_optimization_request(request: Dict[str, Any]) -> bool:
    """
    Validate optimization request.
    
    Args:
        request: Optimization request dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If request is invalid
    """
    required_fields = ["problem_type", "algorithm"]
    
    for field in required_fields:
        if field not in request:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate problem_type
    problem_type = request["problem_type"]
    valid_problem_types = ["5g", "recommendation"]
    if problem_type not in valid_problem_types:
        raise ValidationError(f"Invalid problem_type: {problem_type}. Must be one of {valid_problem_types}")
    
    # Validate algorithm
    algorithm = request["algorithm"]
    valid_algorithms = ["nsga3", "moead", "spea2"]
    if algorithm not in valid_algorithms:
        raise ValidationError(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")
    
    # Validate mode if provided
    if "mode" in request:
        mode = request["mode"]
        valid_modes = ["normal", "emergency", "revenue_focused", "user_experience"]
        if mode not in valid_modes:
            raise ValidationError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
    
    # Validate population_size if provided
    if "population_size" in request:
        population_size = request["population_size"]
        if not isinstance(population_size, int) or population_size <= 0:
            raise ValidationError("population_size must be a positive integer")
        if population_size > 1000:
            raise ValidationError("population_size cannot exceed 1000")
    
    # Validate generations if provided
    if "generations" in request:
        generations = request["generations"]
        if not isinstance(generations, int) or generations <= 0:
            raise ValidationError("generations must be a positive integer")
        if generations > 10000:
            raise ValidationError("generations cannot exceed 10000")
    
    # Validate weights if provided
    if "weights" in request:
        weights = request["weights"]
        if not isinstance(weights, dict):
            raise ValidationError("weights must be a dictionary")
        
        for key, value in weights.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Weight {key} must be a number")
            if value < 0:
                raise ValidationError(f"Weight {key} must be non-negative")
        
        # Check if weights sum to 1.0 (with tolerance)
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValidationError(f"Weights must sum to 1.0, got {total_weight}")
    
    # Validate constraints if provided
    if "constraints" in request:
        constraints = request["constraints"]
        if not isinstance(constraints, dict):
            raise ValidationError("constraints must be a dictionary")
    
    # Validate timeout if provided
    if "timeout" in request:
        timeout = request["timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValidationError("timeout must be a positive number")
        if timeout > 86400:  # 24 hours
            raise ValidationError("timeout cannot exceed 86400 seconds (24 hours)")
    
    return True


def validate_training_request(request: Dict[str, Any]) -> bool:
    """
    Validate training request.
    
    Args:
        request: Training request dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If request is invalid
    """
    required_fields = ["training_type", "algorithm", "dataset", "hyperparameters"]
    
    for field in required_fields:
        if field not in request:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate training_type
    training_type = request["training_type"]
    valid_training_types = ["reinforcement_learning", "deep_learning", "optimization", "hyperparameter_tuning"]
    if training_type not in valid_training_types:
        raise ValidationError(f"Invalid training_type: {training_type}. Must be one of {valid_training_types}")
    
    # Validate algorithm based on training type
    algorithm = request["algorithm"]
    valid_algorithms = {
        "reinforcement_learning": ["sac", "ppo", "dqn", "a3c", "ddpg"],
        "deep_learning": ["adam", "sgd", "rmsprop", "adagrad"],
        "optimization": ["nsga3", "moead", "genetic", "pso"],
        "hyperparameter_tuning": ["optuna", "ray_tune", "hyperopt"]
    }
    
    if algorithm not in valid_algorithms.get(training_type, []):
        raise ValidationError(f"Invalid algorithm {algorithm} for training_type {training_type}")
    
    # Validate dataset
    dataset = request["dataset"]
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValidationError("dataset must be a non-empty string")
    
    # Validate hyperparameters
    hyperparameters = request["hyperparameters"]
    if not isinstance(hyperparameters, dict):
        raise ValidationError("hyperparameters must be a dictionary")
    
    # Validate common hyperparameters
    if "learning_rate" in hyperparameters:
        lr = hyperparameters["learning_rate"]
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            raise ValidationError("learning_rate must be between 0 and 1")
    
    if "batch_size" in hyperparameters:
        batch_size = hyperparameters["batch_size"]
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValidationError("batch_size must be a positive integer")
        if batch_size > 10000:
            raise ValidationError("batch_size cannot exceed 10000")
    
    if "epochs" in hyperparameters:
        epochs = hyperparameters["epochs"]
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValidationError("epochs must be a positive integer")
        if epochs > 10000:
            raise ValidationError("epochs cannot exceed 10000")
    
    # Validate RL-specific hyperparameters
    if training_type == "reinforcement_learning":
        if "total_timesteps" in hyperparameters:
            timesteps = hyperparameters["total_timesteps"]
            if not isinstance(timesteps, int) or timesteps <= 0:
                raise ValidationError("total_timesteps must be a positive integer")
            if timesteps > 10000000:
                raise ValidationError("total_timesteps cannot exceed 10,000,000")
    
    # Validate resources if provided
    if "resources" in request:
        resources = request["resources"]
        if not isinstance(resources, dict):
            raise ValidationError("resources must be a dictionary")
        
        if "cpu" in resources:
            cpu = resources["cpu"]
            if not isinstance(cpu, (int, float)) or cpu <= 0:
                raise ValidationError("cpu resource must be a positive number")
        
        if "memory" in resources:
            memory = resources["memory"]
            if not isinstance(memory, str):
                raise ValidationError("memory resource must be a string (e.g., '4Gi')")
            
            # Validate memory format
            if not re.match(r'^\d+(\.\d+)?(Mi|Gi|Ti)$', memory):
                raise ValidationError("memory format must be like '4Gi', '512Mi', etc.")
        
        if "gpu" in resources:
            gpu = resources["gpu"]
            if not isinstance(gpu, int) or gpu < 0:
                raise ValidationError("gpu resource must be a non-negative integer")
    
    return True


def validate_model_id(model_id: str) -> bool:
    """
    Validate model ID format.
    
    Args:
        model_id: Model identifier
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If model ID is invalid
    """
    if not isinstance(model_id, str):
        raise ValidationError("model_id must be a string")
    
    if not model_id.strip():
        raise ValidationError("model_id cannot be empty")
    
    # Check format: alphanumeric, hyphens, underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', model_id):
        raise ValidationError("model_id can only contain alphanumeric characters, hyphens, and underscores")
    
    if len(model_id) > 100:
        raise ValidationError("model_id cannot exceed 100 characters")
    
    return True


def validate_dataset_id(dataset_id: str) -> bool:
    """
    Validate dataset ID format.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If dataset ID is invalid
    """
    if not isinstance(dataset_id, str):
        raise ValidationError("dataset_id must be a string")
    
    if not dataset_id.strip():
        raise ValidationError("dataset_id cannot be empty")
    
    # Check format: alphanumeric, hyphens, underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', dataset_id):
        raise ValidationError("dataset_id can only contain alphanumeric characters, hyphens, and underscores")
    
    if len(dataset_id) > 100:
        raise ValidationError("dataset_id cannot exceed 100 characters")
    
    return True


def validate_inference_request(request: Dict[str, Any]) -> bool:
    """
    Validate inference request.
    
    Args:
        request: Inference request dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If request is invalid
    """
    required_fields = ["model_id", "inputs"]
    
    for field in required_fields:
        if field not in request:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate model_id
    validate_model_id(request["model_id"])
    
    # Validate inputs
    inputs = request["inputs"]
    if not isinstance(inputs, dict):
        raise ValidationError("inputs must be a dictionary")
    
    if not inputs:
        raise ValidationError("inputs cannot be empty")
    
    # Validate parameters if provided
    if "parameters" in request:
        parameters = request["parameters"]
        if not isinstance(parameters, dict):
            raise ValidationError("parameters must be a dictionary")
    
    return True


def validate_feature_request(request: Dict[str, Any]) -> bool:
    """
    Validate feature request.
    
    Args:
        request: Feature request dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If request is invalid
    """
    required_fields = ["feature_refs", "entity_rows"]
    
    for field in required_fields:
        if field not in request:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate feature_refs
    feature_refs = request["feature_refs"]
    if not isinstance(feature_refs, list):
        raise ValidationError("feature_refs must be a list")
    
    if not feature_refs:
        raise ValidationError("feature_refs cannot be empty")
    
    for ref in feature_refs:
        if not isinstance(ref, str):
            raise ValidationError("Each feature reference must be a string")
    
    # Validate entity_rows
    entity_rows = request["entity_rows"]
    if not isinstance(entity_rows, list):
        raise ValidationError("entity_rows must be a list")
    
    if not entity_rows:
        raise ValidationError("entity_rows cannot be empty")
    
    for row in entity_rows:
        if not isinstance(row, dict):
            raise ValidationError("Each entity row must be a dictionary")
    
    return True


def validate_pagination_params(params: Dict[str, Any]) -> bool:
    """
    Validate pagination parameters.
    
    Args:
        params: Pagination parameters
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if "page" in params:
        page = params["page"]
        if not isinstance(page, int) or page < 1:
            raise ValidationError("page must be a positive integer")
    
    if "page_size" in params:
        page_size = params["page_size"]
        if not isinstance(page_size, int) or page_size < 1:
            raise ValidationError("page_size must be a positive integer")
        if page_size > 1000:
            raise ValidationError("page_size cannot exceed 1000")
    
    if "offset" in params:
        offset = params["offset"]
        if not isinstance(offset, int) or offset < 0:
            raise ValidationError("offset must be a non-negative integer")
    
    if "limit" in params:
        limit = params["limit"]
        if not isinstance(limit, int) or limit < 1:
            raise ValidationError("limit must be a positive integer")
        if limit > 1000:
            raise ValidationError("limit cannot exceed 1000")
    
    return True
