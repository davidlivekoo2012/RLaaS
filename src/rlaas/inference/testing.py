"""
A/B Testing framework for RLaaS inference services.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random
import json
import numpy as np
from scipy import stats

from rlaas.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class ExperimentStatus(Enum):
    """A/B test experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficSplitType(Enum):
    """Traffic split strategies."""
    RANDOM = "random"
    USER_ID_HASH = "user_id_hash"
    GEOGRAPHIC = "geographic"
    CUSTOM = "custom"


class StatisticalTest(Enum):
    """Statistical test methods."""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    BAYESIAN = "bayesian"


@dataclass
class ExperimentVariant:
    """A/B test variant configuration."""
    variant_id: str
    name: str
    description: str
    model_id: str
    model_version: str
    traffic_percentage: float
    
    # Configuration overrides
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    conversion_count: int = 0


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration."""
    experiment_id: str
    name: str
    description: str
    
    # Variants
    variants: List[ExperimentVariant]
    
    # Traffic configuration
    split_type: TrafficSplitType = TrafficSplitType.RANDOM
    split_config: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment settings
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    
    # Success metrics
    primary_metric: str = "conversion_rate"
    secondary_metrics: List[str] = field(default_factory=list)
    
    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentResult:
    """A/B test experiment results."""
    experiment_id: str
    variant_results: Dict[str, Dict[str, Any]]
    statistical_significance: Dict[str, bool]
    confidence_intervals: Dict[str, Dict[str, float]]
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    recommendations: List[str]
    
    # Overall metrics
    total_requests: int = 0
    experiment_duration: timedelta = field(default_factory=lambda: timedelta(0))
    winner_variant: Optional[str] = None


class ABTestManager:
    """
    A/B Testing Manager for inference services.
    
    Provides capabilities for:
    - Experiment design and configuration
    - Traffic splitting and routing
    - Statistical analysis and significance testing
    - Real-time monitoring and alerts
    """
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}
        
        logger.info("ABTestManager initialized")
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new A/B test experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        # Validate configuration
        await self._validate_experiment_config(config)
        
        # Store experiment
        self.experiments[config.experiment_id] = config
        
        logger.info(f"A/B test experiment created: {config.experiment_id}")
        return config.experiment_id
    
    async def _validate_experiment_config(self, config: ExperimentConfig):
        """Validate experiment configuration."""
        
        # Check traffic percentages sum to 100%
        total_traffic = sum(variant.traffic_percentage for variant in config.variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100%, got {total_traffic}")
        
        # Check minimum 2 variants
        if len(config.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        
        # Check variant IDs are unique
        variant_ids = [v.variant_id for v in config.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Variant IDs must be unique")
        
        # Check time configuration
        if config.start_time and config.end_time:
            if config.start_time >= config.end_time:
                raise ValueError("Start time must be before end time")
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an A/B test experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Success status
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment {experiment_id} is not in draft status")
        
        # Set start time if not specified
        if not experiment.start_time:
            experiment.start_time = datetime.now()
        
        # Update status
        experiment.status = ExperimentStatus.RUNNING
        experiment.updated_at = datetime.now()
        
        # Add to active experiments
        self.active_experiments[experiment_id] = experiment
        
        logger.info(f"A/B test experiment started: {experiment_id}")
        return True
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """
        Stop an A/B test experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Success status
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment {experiment_id} is not running")
        
        # Update status
        experiment.status = ExperimentStatus.COMPLETED
        experiment.updated_at = datetime.now()
        experiment.end_time = datetime.now()
        
        # Remove from active experiments
        self.active_experiments.pop(experiment_id, None)
        
        # Generate final results
        await self._generate_experiment_results(experiment_id)
        
        logger.info(f"A/B test experiment stopped: {experiment_id}")
        return True
    
    async def route_request(
        self,
        request_context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Route request to appropriate variant.
        
        Args:
            request_context: Request context (user_id, geo, etc.)
            
        Returns:
            Variant ID or None if no active experiments
        """
        # Find applicable experiments
        applicable_experiments = []
        
        for experiment in self.active_experiments.values():
            if await self._is_experiment_applicable(experiment, request_context):
                applicable_experiments.append(experiment)
        
        if not applicable_experiments:
            return None
        
        # For simplicity, use the first applicable experiment
        # In practice, you might have more complex logic
        experiment = applicable_experiments[0]
        
        # Determine variant based on split strategy
        variant = await self._select_variant(experiment, request_context)
        
        if variant:
            # Record request
            variant.request_count += 1
            
            return variant.variant_id
        
        return None
    
    async def _is_experiment_applicable(
        self,
        experiment: ExperimentConfig,
        request_context: Dict[str, Any]
    ) -> bool:
        """Check if experiment is applicable to request."""
        
        # Check time bounds
        now = datetime.now()
        
        if experiment.start_time and now < experiment.start_time:
            return False
        
        if experiment.end_time and now > experiment.end_time:
            return False
        
        # Check other conditions (could be extended)
        # For now, all running experiments are applicable
        return True
    
    async def _select_variant(
        self,
        experiment: ExperimentConfig,
        request_context: Dict[str, Any]
    ) -> Optional[ExperimentVariant]:
        """Select variant based on split strategy."""
        
        if experiment.split_type == TrafficSplitType.RANDOM:
            return self._select_variant_random(experiment)
        
        elif experiment.split_type == TrafficSplitType.USER_ID_HASH:
            return self._select_variant_user_hash(experiment, request_context)
        
        elif experiment.split_type == TrafficSplitType.GEOGRAPHIC:
            return self._select_variant_geographic(experiment, request_context)
        
        else:
            # Default to random
            return self._select_variant_random(experiment)
    
    def _select_variant_random(self, experiment: ExperimentConfig) -> Optional[ExperimentVariant]:
        """Select variant using random split."""
        
        rand_val = random.random() * 100
        cumulative = 0
        
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if rand_val <= cumulative:
                return variant
        
        # Fallback to first variant
        return experiment.variants[0] if experiment.variants else None
    
    def _select_variant_user_hash(
        self,
        experiment: ExperimentConfig,
        request_context: Dict[str, Any]
    ) -> Optional[ExperimentVariant]:
        """Select variant using user ID hash."""
        
        user_id = request_context.get("user_id")
        if not user_id:
            return self._select_variant_random(experiment)
        
        # Hash user ID to get consistent assignment
        hash_val = hash(f"{experiment.experiment_id}_{user_id}") % 100
        
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if hash_val < cumulative:
                return variant
        
        return experiment.variants[0] if experiment.variants else None
    
    def _select_variant_geographic(
        self,
        experiment: ExperimentConfig,
        request_context: Dict[str, Any]
    ) -> Optional[ExperimentVariant]:
        """Select variant using geographic split."""
        
        # This would implement geographic-based routing
        # For now, fallback to random
        return self._select_variant_random(experiment)
    
    async def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        metrics: Dict[str, Any]
    ):
        """
        Record experiment result.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metrics: Result metrics
        """
        if experiment_id not in self.active_experiments:
            return
        
        experiment = self.active_experiments[experiment_id]
        
        # Find variant
        variant = None
        for v in experiment.variants:
            if v.variant_id == variant_id:
                variant = v
                break
        
        if not variant:
            return
        
        # Update variant metrics
        if metrics.get("success", False):
            variant.success_count += 1
        
        if metrics.get("error", False):
            variant.error_count += 1
        
        if "latency" in metrics:
            variant.total_latency += metrics["latency"]
        
        if metrics.get("conversion", False):
            variant.conversion_count += 1
        
        logger.debug(f"Recorded result for {experiment_id}/{variant_id}: {metrics}")
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment status and metrics."""
        
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        # Calculate metrics for each variant
        variant_metrics = {}
        for variant in experiment.variants:
            avg_latency = (variant.total_latency / variant.request_count 
                          if variant.request_count > 0 else 0)
            
            conversion_rate = (variant.conversion_count / variant.request_count 
                             if variant.request_count > 0 else 0)
            
            error_rate = (variant.error_count / variant.request_count 
                         if variant.request_count > 0 else 0)
            
            variant_metrics[variant.variant_id] = {
                "request_count": variant.request_count,
                "success_count": variant.success_count,
                "error_count": variant.error_count,
                "conversion_count": variant.conversion_count,
                "avg_latency": avg_latency,
                "conversion_rate": conversion_rate,
                "error_rate": error_rate
            }
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
            "variants": variant_metrics,
            "total_requests": sum(v.request_count for v in experiment.variants)
        }
    
    async def _generate_experiment_results(self, experiment_id: str):
        """Generate statistical analysis results."""
        
        experiment = self.experiments[experiment_id]
        
        # Calculate variant results
        variant_results = {}
        for variant in experiment.variants:
            conversion_rate = (variant.conversion_count / variant.request_count 
                             if variant.request_count > 0 else 0)
            
            avg_latency = (variant.total_latency / variant.request_count 
                          if variant.request_count > 0 else 0)
            
            variant_results[variant.variant_id] = {
                "conversion_rate": conversion_rate,
                "avg_latency": avg_latency,
                "sample_size": variant.request_count,
                "conversions": variant.conversion_count
            }
        
        # Perform statistical tests
        statistical_significance = {}
        p_values = {}
        confidence_intervals = {}
        effect_sizes = {}
        
        if len(experiment.variants) >= 2:
            control_variant = experiment.variants[0]
            
            for i, variant in enumerate(experiment.variants[1:], 1):
                comparison_key = f"{control_variant.variant_id}_vs_{variant.variant_id}"
                
                # Perform t-test for conversion rates
                if (control_variant.request_count > 0 and variant.request_count > 0):
                    
                    # Create binary arrays for conversion
                    control_conversions = np.array([1] * control_variant.conversion_count + 
                                                 [0] * (control_variant.request_count - control_variant.conversion_count))
                    variant_conversions = np.array([1] * variant.conversion_count + 
                                                 [0] * (variant.request_count - variant.conversion_count))
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(control_conversions, variant_conversions)
                    
                    statistical_significance[comparison_key] = p_value < (1 - experiment.confidence_level)
                    p_values[comparison_key] = p_value
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(control_conversions) - 1) * np.var(control_conversions) + 
                                        (len(variant_conversions) - 1) * np.var(variant_conversions)) / 
                                       (len(control_conversions) + len(variant_conversions) - 2))
                    
                    if pooled_std > 0:
                        effect_size = (np.mean(variant_conversions) - np.mean(control_conversions)) / pooled_std
                        effect_sizes[comparison_key] = effect_size
        
        # Generate recommendations
        recommendations = []
        
        # Find best performing variant
        best_variant = None
        best_conversion_rate = -1
        
        for variant_id, results in variant_results.items():
            if results["conversion_rate"] > best_conversion_rate:
                best_conversion_rate = results["conversion_rate"]
                best_variant = variant_id
        
        if best_variant:
            recommendations.append(f"Variant {best_variant} shows the highest conversion rate ({best_conversion_rate:.3f})")
        
        # Check for statistical significance
        significant_improvements = [k for k, v in statistical_significance.items() if v]
        if significant_improvements:
            recommendations.append(f"Statistically significant improvements found: {', '.join(significant_improvements)}")
        else:
            recommendations.append("No statistically significant differences found")
        
        # Create results object
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            confidence_intervals=confidence_intervals,
            p_values=p_values,
            effect_sizes=effect_sizes,
            recommendations=recommendations,
            total_requests=sum(v.request_count for v in experiment.variants),
            experiment_duration=experiment.end_time - experiment.start_time if experiment.end_time and experiment.start_time else timedelta(0),
            winner_variant=best_variant
        )
        
        self.experiment_results[experiment_id] = result
        
        logger.info(f"Generated results for experiment {experiment_id}")
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment results."""
        return self.experiment_results.get(experiment_id)
    
    async def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List experiments with optional status filter."""
        
        experiments = []
        for experiment in self.experiments.values():
            if status is None or experiment.status == status:
                experiments.append({
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "status": experiment.status.value,
                    "variants_count": len(experiment.variants),
                    "created_at": experiment.created_at.isoformat(),
                    "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
                    "end_time": experiment.end_time.isoformat() if experiment.end_time else None
                })
        
        return experiments
