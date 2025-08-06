"""
Distributed Training support for RLaaS platform.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from rlaas.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class DistributedBackend(Enum):
    """Distributed training backends."""
    PYTORCH_DDP = "pytorch_ddp"
    HOROVOD = "horovod"
    DEEPSPEED = "deepspeed"
    RAY = "ray"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: DistributedBackend
    world_size: int
    num_nodes: int = 1
    gpus_per_node: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # Backend-specific configs
    horovod_config: Dict[str, Any] = field(default_factory=dict)
    deepspeed_config: Dict[str, Any] = field(default_factory=dict)
    ray_config: Dict[str, Any] = field(default_factory=dict)


class DistributedTrainer:
    """
    Distributed training coordinator.
    
    Supports multiple distributed training backends:
    - PyTorch DDP (Distributed Data Parallel)
    - Horovod
    - DeepSpeed
    - Ray Train
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.local_rank = 0
        self.global_rank = 0
        
        logger.info(f"DistributedTrainer initialized with {config.backend.value}")
    
    async def initialize(self):
        """Initialize distributed training environment."""
        
        if self.is_initialized:
            return
        
        if self.config.backend == DistributedBackend.PYTORCH_DDP:
            await self._init_pytorch_ddp()
        elif self.config.backend == DistributedBackend.HOROVOD:
            await self._init_horovod()
        elif self.config.backend == DistributedBackend.DEEPSPEED:
            await self._init_deepspeed()
        elif self.config.backend == DistributedBackend.RAY:
            await self._init_ray()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        
        self.is_initialized = True
        logger.info(f"Distributed training initialized: rank {self.global_rank}/{self.config.world_size}")
    
    async def _init_pytorch_ddp(self):
        """Initialize PyTorch DDP."""
        
        # Get environment variables set by launcher
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=world_size,
                rank=self.global_rank
            )
        
        logger.info(f"PyTorch DDP initialized: rank {self.global_rank}, device {device}")
    
    async def _init_horovod(self):
        """Initialize Horovod."""
        
        try:
            import horovod.torch as hvd
            
            hvd.init()
            self.local_rank = hvd.local_rank()
            self.global_rank = hvd.rank()
            
            # Set CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            
            logger.info(f"Horovod initialized: rank {self.global_rank}/{hvd.size()}")
            
        except ImportError:
            raise RuntimeError("Horovod not installed. Install with: pip install horovod")
    
    async def _init_deepspeed(self):
        """Initialize DeepSpeed."""
        
        try:
            import deepspeed
            
            # DeepSpeed initialization is handled during model creation
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.global_rank = int(os.environ.get("RANK", 0))
            
            logger.info(f"DeepSpeed environment prepared: rank {self.global_rank}")
            
        except ImportError:
            raise RuntimeError("DeepSpeed not installed. Install with: pip install deepspeed")
    
    async def _init_ray(self):
        """Initialize Ray Train."""
        
        try:
            import ray
            from ray import train
            
            if not ray.is_initialized():
                ray.init(address="auto")  # Connect to existing cluster or start local
            
            # Ray Train context
            train_context = train.get_context()
            self.local_rank = train_context.get_local_rank()
            self.global_rank = train_context.get_world_rank()
            
            logger.info(f"Ray Train initialized: rank {self.global_rank}")
            
        except ImportError:
            raise RuntimeError("Ray not installed. Install with: pip install ray[train]")
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        
        if not self.is_initialized:
            raise RuntimeError("Distributed trainer not initialized")
        
        if self.config.backend == DistributedBackend.PYTORCH_DDP:
            return self._wrap_pytorch_ddp(model)
        elif self.config.backend == DistributedBackend.HOROVOD:
            return self._wrap_horovod(model)
        elif self.config.backend == DistributedBackend.DEEPSPEED:
            return self._wrap_deepspeed(model)
        elif self.config.backend == DistributedBackend.RAY:
            return self._wrap_ray(model)
        else:
            return model
    
    def _wrap_pytorch_ddp(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model with PyTorch DDP."""
        
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if torch.cuda.device_count() > 1:
            model = DDP(model, device_ids=[self.local_rank])
        
        return model
    
    def _wrap_horovod(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for Horovod."""
        
        import horovod.torch as hvd
        
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model
    
    def _wrap_deepspeed(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model with DeepSpeed."""
        
        import deepspeed
        
        # DeepSpeed config
        ds_config = {
            "train_batch_size": self.config.deepspeed_config.get("train_batch_size", 16),
            "gradient_accumulation_steps": self.config.deepspeed_config.get("gradient_accumulation_steps", 1),
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": self.config.deepspeed_config.get("learning_rate", 3e-4)
                }
            },
            "fp16": {
                "enabled": self.config.deepspeed_config.get("fp16", False)
            },
            "zero_optimization": {
                "stage": self.config.deepspeed_config.get("zero_stage", 1)
            }
        }
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        
        return model_engine
    
    def _wrap_ray(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for Ray Train."""
        
        from ray.train.torch import prepare_model
        
        return prepare_model(model)
    
    def wrap_optimizer(self, optimizer, model: torch.nn.Module):
        """Wrap optimizer for distributed training."""
        
        if self.config.backend == DistributedBackend.HOROVOD:
            import horovod.torch as hvd
            return hvd.DistributedOptimizer(
                optimizer, 
                named_parameters=model.named_parameters()
            )
        elif self.config.backend == DistributedBackend.RAY:
            from ray.train.torch import prepare_optimizer
            return prepare_optimizer(optimizer)
        else:
            return optimizer
    
    def wrap_dataloader(self, dataloader):
        """Wrap dataloader for distributed training."""
        
        if self.config.backend == DistributedBackend.PYTORCH_DDP:
            from torch.utils.data.distributed import DistributedSampler
            
            # Create distributed sampler
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.config.world_size,
                rank=self.global_rank
            )
            
            # Create new dataloader with distributed sampler
            return torch.utils.data.DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=sampler,
                num_workers=dataloader.num_workers,
                pin_memory=dataloader.pin_memory
            )
        
        elif self.config.backend == DistributedBackend.RAY:
            from ray.train.torch import prepare_data_loader
            return prepare_data_loader(dataloader)
        
        else:
            return dataloader
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce operation across all processes."""
        
        if self.config.backend == DistributedBackend.PYTORCH_DDP:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.config.world_size
        
        elif self.config.backend == DistributedBackend.HOROVOD:
            import horovod.torch as hvd
            tensor = hvd.allreduce(tensor, average=True)
        
        return tensor
    
    def barrier(self):
        """Synchronization barrier."""
        
        if self.config.backend == DistributedBackend.PYTORCH_DDP:
            dist.barrier()
        elif self.config.backend == DistributedBackend.HOROVOD:
            import horovod.torch as hvd
            hvd.allreduce(torch.tensor(0.0), average=True)
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.global_rank == 0
    
    def cleanup(self):
        """Clean up distributed training."""
        
        if self.config.backend == DistributedBackend.PYTORCH_DDP:
            if dist.is_initialized():
                dist.destroy_process_group()
        
        elif self.config.backend == DistributedBackend.RAY:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        
        self.is_initialized = False
        logger.info("Distributed training cleaned up")


class DistributedLauncher:
    """Launcher for distributed training jobs."""
    
    @staticmethod
    async def launch_pytorch_ddp(
        script_path: str,
        config: DistributedConfig,
        script_args: List[str] = None
    ) -> subprocess.Popen:
        """Launch PyTorch DDP training."""
        
        cmd = [
            "python", "-m", "torch.distributed.launch",
            f"--nproc_per_node={config.gpus_per_node}",
            f"--nnodes={config.num_nodes}",
            f"--master_addr={config.master_addr}",
            f"--master_port={config.master_port}",
            script_path
        ]
        
        if script_args:
            cmd.extend(script_args)
        
        logger.info(f"Launching PyTorch DDP: {' '.join(cmd)}")
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    @staticmethod
    async def launch_horovod(
        script_path: str,
        config: DistributedConfig,
        script_args: List[str] = None
    ) -> subprocess.Popen:
        """Launch Horovod training."""
        
        cmd = [
            "horovodrun",
            f"-np {config.world_size}",
            f"-H localhost:{config.gpus_per_node}",
            "python", script_path
        ]
        
        if script_args:
            cmd.extend(script_args)
        
        logger.info(f"Launching Horovod: {' '.join(cmd)}")
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
    
    @staticmethod
    async def launch_ray(
        script_path: str,
        config: DistributedConfig,
        script_args: List[str] = None
    ) -> str:
        """Launch Ray Train job."""
        
        import ray
        from ray import train
        from ray.train import ScalingConfig
        
        if not ray.is_initialized():
            ray.init(address="auto")
        
        # Create scaling config
        scaling_config = ScalingConfig(
            num_workers=config.world_size,
            use_gpu=config.gpus_per_node > 0
        )
        
        # This would typically be a Ray Train Trainer
        # For now, return a job ID placeholder
        job_id = f"ray_train_{hash(script_path)}"
        
        logger.info(f"Ray Train job submitted: {job_id}")
        
        return job_id
