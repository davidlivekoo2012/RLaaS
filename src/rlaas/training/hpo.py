"""
Hyperparameter Optimization Engine for RLaaS platform.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from rlaas.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class HPOBackend(Enum):
    """HPO backend options."""
    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"
    HYPEROPT = "hyperopt"


class SamplerType(Enum):
    """Sampler types for HPO."""
    TPE = "tpe"
    RANDOM = "random"
    CMA_ES = "cma_es"
    GRID = "grid"


class PrunerType(Enum):
    """Pruner types for early stopping."""
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    NONE = "none"


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    backend: HPOBackend = HPOBackend.OPTUNA
    sampler: SamplerType = SamplerType.TPE
    pruner: PrunerType = PrunerType.MEDIAN
    
    # Study configuration
    study_name: str = "rlaas_hpo_study"
    direction: str = "maximize"  # maximize or minimize
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    
    # Search space
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Backend-specific configs
    optuna_config: Dict[str, Any] = field(default_factory=dict)
    ray_tune_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HPOTrial:
    """HPO trial result."""
    trial_id: str
    params: Dict[str, Any]
    value: float
    state: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    intermediate_values: List[float] = field(default_factory=list)


class HPOEngine:
    """
    Hyperparameter Optimization Engine.
    
    Supports multiple HPO backends:
    - Optuna
    - Ray Tune
    - Hyperopt
    """
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.study = None
        self.trials: Dict[str, HPOTrial] = {}
        
        logger.info(f"HPOEngine initialized with {config.backend.value}")
    
    async def create_study(self) -> str:
        """Create HPO study."""
        
        if self.config.backend == HPOBackend.OPTUNA:
            return await self._create_optuna_study()
        elif self.config.backend == HPOBackend.RAY_TUNE:
            return await self._create_ray_tune_study()
        else:
            raise ValueError(f"Unsupported HPO backend: {self.config.backend}")
    
    async def _create_optuna_study(self) -> str:
        """Create Optuna study."""
        
        # Create sampler
        if self.config.sampler == SamplerType.TPE:
            sampler = TPESampler(**self.config.optuna_config.get("sampler_kwargs", {}))
        elif self.config.sampler == SamplerType.RANDOM:
            sampler = RandomSampler()
        elif self.config.sampler == SamplerType.CMA_ES:
            sampler = CmaEsSampler()
        else:
            sampler = TPESampler()
        
        # Create pruner
        if self.config.pruner == PrunerType.MEDIAN:
            pruner = MedianPruner(**self.config.optuna_config.get("pruner_kwargs", {}))
        elif self.config.pruner == PrunerType.HYPERBAND:
            pruner = HyperbandPruner(**self.config.optuna_config.get("pruner_kwargs", {}))
        else:
            pruner = None
        
        # Create study
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.config.optuna_config.get("storage"),
            load_if_exists=True
        )
        
        logger.info(f"Optuna study created: {self.config.study_name}")
        return self.config.study_name
    
    async def _create_ray_tune_study(self) -> str:
        """Create Ray Tune study."""
        
        if not ray.is_initialized():
            ray.init(address="auto")
        
        logger.info(f"Ray Tune study prepared: {self.config.study_name}")
        return self.config.study_name
    
    async def optimize(
        self,
        objective_function: Callable,
        search_space: Optional[Dict[str, Any]] = None
    ) -> List[HPOTrial]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_function: Function to optimize
            search_space: Parameter search space
            
        Returns:
            List of trials
        """
        search_space = search_space or self.config.search_space
        
        if self.config.backend == HPOBackend.OPTUNA:
            return await self._optimize_optuna(objective_function, search_space)
        elif self.config.backend == HPOBackend.RAY_TUNE:
            return await self._optimize_ray_tune(objective_function, search_space)
        else:
            raise ValueError(f"Unsupported HPO backend: {self.config.backend}")
    
    async def _optimize_optuna(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any]
    ) -> List[HPOTrial]:
        """Run Optuna optimization."""
        
        def optuna_objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
            
            # Run objective function
            try:
                result = objective_function(params)
                
                # Handle intermediate values for pruning
                if isinstance(result, dict) and "intermediate_values" in result:
                    for step, value in enumerate(result["intermediate_values"]):
                        trial.report(value, step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    return result["value"]
                else:
                    return result
                    
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                raise optuna.TrialPruned()
        
        # Run optimization
        self.study.optimize(
            optuna_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        # Convert trials to our format
        trials = []
        for trial in self.study.trials:
            hpo_trial = HPOTrial(
                trial_id=str(trial.number),
                params=trial.params,
                value=trial.value if trial.value is not None else float('-inf'),
                state=trial.state.name,
                start_time=trial.datetime_start.isoformat() if trial.datetime_start else None,
                end_time=trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                intermediate_values=list(trial.intermediate_values.values())
            )
            trials.append(hpo_trial)
            self.trials[hpo_trial.trial_id] = hpo_trial
        
        logger.info(f"Optuna optimization completed: {len(trials)} trials")
        return trials
    
    async def _optimize_ray_tune(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any]
    ) -> List[HPOTrial]:
        """Run Ray Tune optimization."""
        
        # Convert search space to Ray Tune format
        tune_search_space = {}
        for param_name, param_config in search_space.items():
            if param_config["type"] == "float":
                if param_config.get("log", False):
                    tune_search_space[param_name] = tune.loguniform(
                        param_config["low"], param_config["high"]
                    )
                else:
                    tune_search_space[param_name] = tune.uniform(
                        param_config["low"], param_config["high"]
                    )
            elif param_config["type"] == "int":
                tune_search_space[param_name] = tune.randint(
                    param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "categorical":
                tune_search_space[param_name] = tune.choice(param_config["choices"])
        
        # Create scheduler
        scheduler = ASHAScheduler(
            metric="score",
            mode="max" if self.config.direction == "maximize" else "min",
            max_t=self.config.ray_tune_config.get("max_t", 100),
            grace_period=self.config.ray_tune_config.get("grace_period", 10)
        )
        
        # Wrap objective function for Ray Tune
        def tune_objective(config_dict):
            result = objective_function(config_dict)
            
            if isinstance(result, dict):
                tune.report(**result)
            else:
                tune.report(score=result)
        
        # Run optimization
        analysis = tune.run(
            tune_objective,
            config=tune_search_space,
            num_samples=self.config.n_trials,
            scheduler=scheduler,
            time_budget_s=self.config.timeout,
            name=self.config.study_name
        )
        
        # Convert results to our format
        trials = []
        for trial_id, trial_result in analysis.results.items():
            hpo_trial = HPOTrial(
                trial_id=trial_id,
                params=trial_result.config,
                value=trial_result.get("score", float('-inf')),
                state="COMPLETE" if trial_result.get("done", False) else "RUNNING"
            )
            trials.append(hpo_trial)
            self.trials[hpo_trial.trial_id] = hpo_trial
        
        logger.info(f"Ray Tune optimization completed: {len(trials)} trials")
        return trials
    
    async def get_best_trial(self) -> Optional[HPOTrial]:
        """Get best trial."""
        
        if self.config.backend == HPOBackend.OPTUNA and self.study:
            best_trial = self.study.best_trial
            return HPOTrial(
                trial_id=str(best_trial.number),
                params=best_trial.params,
                value=best_trial.value,
                state=best_trial.state.name
            )
        
        elif self.trials:
            # Find best trial manually
            best_trial = None
            best_value = float('-inf') if self.config.direction == "maximize" else float('inf')
            
            for trial in self.trials.values():
                if self.config.direction == "maximize":
                    if trial.value > best_value:
                        best_value = trial.value
                        best_trial = trial
                else:
                    if trial.value < best_value:
                        best_value = trial.value
                        best_trial = trial
            
            return best_trial
        
        return None
    
    async def get_trial_history(self) -> List[HPOTrial]:
        """Get trial history."""
        return list(self.trials.values())
    
    async def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest next parameters to try."""
        
        if self.config.backend == HPOBackend.OPTUNA and self.study:
            trial = self.study.ask()
            
            params = {}
            for param_name, param_config in self.config.search_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
            
            return params
        
        else:
            # Random sampling fallback
            import random
            params = {}
            for param_name, param_config in self.config.search_space.items():
                if param_config["type"] == "float":
                    if param_config.get("log", False):
                        import math
                        log_low = math.log(param_config["low"])
                        log_high = math.log(param_config["high"])
                        params[param_name] = math.exp(random.uniform(log_low, log_high))
                    else:
                        params[param_name] = random.uniform(
                            param_config["low"], param_config["high"]
                        )
                elif param_config["type"] == "int":
                    params[param_name] = random.randint(
                        param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = random.choice(param_config["choices"])
            
            return params
    
    async def report_result(self, trial_id: str, value: float, intermediate: bool = False):
        """Report trial result."""
        
        if trial_id in self.trials:
            trial = self.trials[trial_id]
            if intermediate:
                trial.intermediate_values.append(value)
            else:
                trial.value = value
                trial.state = "COMPLETE"
        
        logger.info(f"Trial {trial_id} result reported: {value}")
    
    async def get_study_statistics(self) -> Dict[str, Any]:
        """Get study statistics."""
        
        if not self.trials:
            return {"n_trials": 0}
        
        completed_trials = [t for t in self.trials.values() if t.state == "COMPLETE"]
        
        if not completed_trials:
            return {"n_trials": len(self.trials), "n_completed": 0}
        
        values = [t.value for t in completed_trials]
        
        return {
            "n_trials": len(self.trials),
            "n_completed": len(completed_trials),
            "best_value": max(values) if self.config.direction == "maximize" else min(values),
            "mean_value": sum(values) / len(values),
            "std_value": (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5
        }
