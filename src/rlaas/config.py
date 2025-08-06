"""
Configuration management for RLaaS platform.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    url: str = Field(
        default="postgresql://rlaas:rlaas@localhost:5432/rlaas",
        env="DATABASE_URL"
    )
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")


class RedisConfig(BaseSettings):
    """Redis configuration."""
    
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=30, env="REDIS_SOCKET_TIMEOUT")


class KafkaConfig(BaseSettings):
    """Kafka configuration."""
    
    bootstrap_servers: List[str] = Field(
        default=["localhost:9092"], 
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    security_protocol: str = Field(default="PLAINTEXT", env="KAFKA_SECURITY_PROTOCOL")
    sasl_mechanism: Optional[str] = Field(default=None, env="KAFKA_SASL_MECHANISM")
    sasl_username: Optional[str] = Field(default=None, env="KAFKA_SASL_USERNAME")
    sasl_password: Optional[str] = Field(default=None, env="KAFKA_SASL_PASSWORD")


class MLflowConfig(BaseSettings):
    """MLflow configuration."""
    
    tracking_uri: str = Field(
        default="http://localhost:5000", 
        env="MLFLOW_TRACKING_URI"
    )
    experiment_name: str = Field(default="rlaas", env="MLFLOW_EXPERIMENT_NAME")
    artifact_location: Optional[str] = Field(
        default=None, 
        env="MLFLOW_ARTIFACT_LOCATION"
    )


class OptimizationConfig(BaseSettings):
    """Multi-objective optimization configuration."""
    
    nsga_population_size: int = Field(default=100, env="NSGA_POPULATION_SIZE")
    nsga_generations: int = Field(default=500, env="NSGA_GENERATIONS")
    moea_decomposition_method: str = Field(
        default="tchebycheff", 
        env="MOEA_DECOMPOSITION_METHOD"
    )
    topsis_weights_update_interval: int = Field(
        default=300, 
        env="TOPSIS_WEIGHTS_UPDATE_INTERVAL"
    )
    pareto_frontier_size: int = Field(default=50, env="PARETO_FRONTIER_SIZE")


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30, 
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    refresh_token_expire_days: int = Field(
        default=7, 
        env="REFRESH_TOKEN_EXPIRE_DAYS"
    )


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""
    
    prometheus_port: int = Field(default=8090, env="PROMETHEUS_PORT")
    jaeger_agent_host: str = Field(default="localhost", env="JAEGER_AGENT_HOST")
    jaeger_agent_port: int = Field(default=6831, env="JAEGER_AGENT_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")


class Config(BaseSettings):
    """Main configuration class."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Components
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    kafka: KafkaConfig = KafkaConfig()
    mlflow: MLflowConfig = MLflowConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
