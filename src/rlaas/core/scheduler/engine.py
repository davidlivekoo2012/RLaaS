"""
Adaptive Scheduler Engine for dynamic resource allocation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
from datetime import datetime, timedelta

from .resource_manager import ResourceManager
from .load_balancer import LoadBalancer
from .priority_manager import PriorityManager
from rlaas.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class TaskType(Enum):
    """Types of tasks that can be scheduled."""
    OPTIMIZATION = "optimization"
    TRAINING = "training"
    INFERENCE = "inference"
    DATA_PROCESSING = "data_processing"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a task to be scheduled."""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    resource_requirements: Dict[str, Any]
    estimated_duration: float  # seconds
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime information
    status: TaskStatus = TaskStatus.PENDING
    assigned_resources: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actual_duration: Optional[float] = None


@dataclass
class SchedulingDecision:
    """Represents a scheduling decision."""
    task_id: str
    assigned_resources: Dict[str, Any]
    scheduled_time: datetime
    estimated_completion: datetime
    reasoning: str


class AdaptiveScheduler:
    """
    Adaptive scheduler for dynamic resource allocation and load balancing.
    
    Uses machine learning to predict resource needs and optimize scheduling
    decisions based on historical performance and current system state.
    """
    
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.load_balancer = LoadBalancer()
        self.priority_manager = PriorityManager()
        
        # Task management
        self.pending_tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        
        # Scheduling state
        self.scheduling_interval = 5.0  # seconds
        self.is_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.scheduling_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        logger.info("AdaptiveScheduler initialized")
    
    async def start(self):
        """Start the adaptive scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        logger.info("Adaptive scheduler started")
    
    async def stop(self):
        """Stop the adaptive scheduler."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Adaptive scheduler stopped")
    
    async def submit_task(self, task: Task) -> str:
        """
        Submit a task for scheduling.
        
        Args:
            task: Task to be scheduled
            
        Returns:
            Task ID
        """
        self.pending_tasks[task.task_id] = task
        
        # Update priority based on current system state
        await self.priority_manager.update_task_priority(task)
        
        logger.info(f"Task {task.task_id} submitted for scheduling")
        return task.task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled successfully
        """
        # Check pending tasks
        if task_id in self.pending_tasks:
            task = self.pending_tasks.pop(task_id)
            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now()
            self.completed_tasks[task_id] = task
            logger.info(f"Pending task {task_id} cancelled")
            return True
        
        # Check running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks.pop(task_id)
            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now()
            
            # Release resources
            if task.assigned_resources:
                await self.resource_manager.release_resources(task.assigned_resources)
            
            self.completed_tasks[task_id] = task
            logger.info(f"Running task {task_id} cancelled")
            return True
        
        logger.warning(f"Task {task_id} not found for cancellation")
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        
        # Check all task collections
        for task_dict, status in [
            (self.pending_tasks, "pending"),
            (self.running_tasks, "running"),
            (self.completed_tasks, "completed")
        ]:
            if task_id in task_dict:
                task = task_dict[task_id]
                return {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "start_time": task.start_time.isoformat() if task.start_time else None,
                    "end_time": task.end_time.isoformat() if task.end_time else None,
                    "estimated_duration": task.estimated_duration,
                    "actual_duration": task.actual_duration,
                    "assigned_resources": task.assigned_resources,
                    "metadata": task.metadata
                }
        
        return None
    
    async def _scheduling_loop(self):
        """Main scheduling loop."""
        
        while self.is_running:
            try:
                await self._schedule_tasks()
                await self._monitor_running_tasks()
                await self._update_performance_metrics()
                
                await asyncio.sleep(self.scheduling_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _schedule_tasks(self):
        """Schedule pending tasks based on available resources and priorities."""
        
        if not self.pending_tasks:
            return
        
        # Get current resource availability
        available_resources = await self.resource_manager.get_available_resources()
        
        # Sort tasks by priority and deadline
        sorted_tasks = sorted(
            self.pending_tasks.values(),
            key=lambda t: (
                -t.priority.value,  # Higher priority first
                t.deadline or datetime.max,  # Earlier deadline first
                t.estimated_duration  # Shorter tasks first
            )
        )
        
        scheduled_tasks = []
        
        for task in sorted_tasks:
            # Check if task dependencies are satisfied
            if not await self._check_dependencies(task):
                continue
            
            # Try to allocate resources
            allocation = await self.resource_manager.allocate_resources(
                task.resource_requirements,
                available_resources
            )
            
            if allocation:
                # Schedule the task
                decision = SchedulingDecision(
                    task_id=task.task_id,
                    assigned_resources=allocation,
                    scheduled_time=datetime.now(),
                    estimated_completion=datetime.now() + timedelta(seconds=task.estimated_duration),
                    reasoning=f"Priority: {task.priority.value}, Resources available"
                )
                
                await self._execute_scheduling_decision(decision)
                scheduled_tasks.append(task.task_id)
                
                # Update available resources
                available_resources = await self.resource_manager.get_available_resources()
        
        if scheduled_tasks:
            logger.info(f"Scheduled {len(scheduled_tasks)} tasks: {scheduled_tasks}")
    
    async def _execute_scheduling_decision(self, decision: SchedulingDecision):
        """Execute a scheduling decision."""
        
        task_id = decision.task_id
        task = self.pending_tasks.pop(task_id)
        
        # Update task information
        task.status = TaskStatus.RUNNING
        task.assigned_resources = decision.assigned_resources
        task.start_time = decision.scheduled_time
        
        # Move to running tasks
        self.running_tasks[task_id] = task
        
        # Record scheduling decision
        self.scheduling_history.append({
            "timestamp": decision.scheduled_time.isoformat(),
            "task_id": task_id,
            "decision": decision.__dict__,
            "system_load": await self.load_balancer.get_current_load()
        })
        
        # Start task execution (this would integrate with actual execution systems)
        asyncio.create_task(self._simulate_task_execution(task))
        
        logger.info(f"Task {task_id} started execution")
    
    async def _simulate_task_execution(self, task: Task):
        """
        Simulate task execution.
        
        In a real implementation, this would integrate with actual
        execution systems like Kubernetes, Docker, or cloud services.
        """
        
        try:
            # Add some randomness to execution time
            actual_duration = task.estimated_duration * np.random.uniform(0.8, 1.2)
            await asyncio.sleep(min(actual_duration, 10.0))  # Cap for simulation
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            task.actual_duration = actual_duration
            
            # Release resources
            if task.assigned_resources:
                await self.resource_manager.release_resources(task.assigned_resources)
            
            # Move to completed tasks
            self.running_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = task
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Mark task as failed
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            
            # Release resources
            if task.assigned_resources:
                await self.resource_manager.release_resources(task.assigned_resources)
            
            # Move to completed tasks
            self.running_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = task
            
            logger.error(f"Task {task.task_id} failed: {e}")
    
    async def _monitor_running_tasks(self):
        """Monitor running tasks for timeouts and resource usage."""
        
        current_time = datetime.now()
        
        for task_id, task in list(self.running_tasks.items()):
            # Check for timeout
            if task.deadline and current_time > task.deadline:
                logger.warning(f"Task {task_id} exceeded deadline")
                await self.cancel_task(task_id)
                continue
            
            # Check for resource usage anomalies
            if task.assigned_resources:
                usage = await self.resource_manager.get_resource_usage(task.assigned_resources)
                if usage and usage.get("cpu_percent", 0) > 95:
                    logger.warning(f"Task {task_id} using high CPU: {usage['cpu_percent']:.1f}%")
    
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        
        for dep_task_id in task.dependencies:
            if dep_task_id in self.pending_tasks or dep_task_id in self.running_tasks:
                return False
            
            if dep_task_id in self.completed_tasks:
                dep_task = self.completed_tasks[dep_task_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
            else:
                # Dependency not found
                return False
        
        return True
    
    async def _update_performance_metrics(self):
        """Update scheduler performance metrics."""
        
        if not self.completed_tasks:
            return
        
        completed_tasks = list(self.completed_tasks.values())
        
        # Calculate metrics
        successful_tasks = [t for t in completed_tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in completed_tasks if t.status == TaskStatus.FAILED]
        
        success_rate = len(successful_tasks) / len(completed_tasks) if completed_tasks else 0
        
        # Average execution time accuracy
        time_accuracies = []
        for task in successful_tasks:
            if task.actual_duration and task.estimated_duration:
                accuracy = 1.0 - abs(task.actual_duration - task.estimated_duration) / task.estimated_duration
                time_accuracies.append(max(0, accuracy))
        
        avg_time_accuracy = np.mean(time_accuracies) if time_accuracies else 0
        
        # Resource utilization
        current_utilization = await self.resource_manager.get_overall_utilization()
        
        # Update metrics
        self.performance_metrics.update({
            "success_rate": success_rate,
            "avg_time_accuracy": avg_time_accuracy,
            "resource_utilization": current_utilization,
            "total_tasks": len(completed_tasks),
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "failed_tasks": len(failed_tasks),
            "last_updated": time.time()
        })
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        
        resource_status = await self.resource_manager.get_status()
        load_status = await self.load_balancer.get_status()
        
        return {
            "scheduler": {
                "is_running": self.is_running,
                "pending_tasks": len(self.pending_tasks),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
            },
            "resources": resource_status,
            "load_balancer": load_status,
            "performance": self.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
