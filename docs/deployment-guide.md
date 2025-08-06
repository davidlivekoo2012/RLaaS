# RLaaS Deployment Guide

This guide provides comprehensive instructions for deploying the RLaaS platform in different environments.

## 🎯 **Deployment Options**

### 1. **Local Development**
- Docker Compose for quick local setup
- All services running on localhost
- Suitable for development and testing

### 2. **Kubernetes (Production)**
- Full production deployment on Kubernetes
- High availability and scalability
- Monitoring and observability included

### 3. **Cloud Platforms**
- AWS EKS, Google GKE, Azure AKS
- Cloud-native services integration
- Auto-scaling and managed services

## 🚀 **Quick Start - Local Development**

### Prerequisites
- Docker Desktop 4.0+
- Docker Compose 2.0+
- 8GB+ RAM available
- 20GB+ disk space

### Steps

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd rlaas
   cp .env.example .env
   ```

2. **Start Development Environment**
   ```bash
   # Linux/Mac
   ./scripts/start-dev.sh
   
   # Windows
   .\scripts\start-dev.ps1
   ```

3. **Verify Deployment**
   ```bash
   # Check health
   curl http://localhost:8000/health
   
   # Access services
   # API Gateway: http://localhost:8000
   # Web Console: http://localhost:8080
   # MLflow: http://localhost:5000
   # Grafana: http://localhost:3000
   ```

## ☸️ **Kubernetes Deployment**

### Prerequisites
- Kubernetes cluster 1.28+
- kubectl configured
- Helm 3.0+
- 16GB+ RAM per node
- GPU nodes (optional, for RL training)

### Option 1: Using Helm Chart (Recommended)

1. **Add Helm Repository**
   ```bash
   helm repo add rlaas https://charts.rlaas.ai
   helm repo update
   ```

2. **Install RLaaS**
   ```bash
   # Create namespace
   kubectl create namespace rlaas-system
   
   # Install with default values
   helm install rlaas rlaas/rlaas -n rlaas-system
   
   # Or with custom values
   helm install rlaas rlaas/rlaas -n rlaas-system -f values-production.yaml
   ```

3. **Verify Installation**
   ```bash
   kubectl get pods -n rlaas-system
   kubectl get services -n rlaas-system
   ```

### Option 2: Using Kustomize

1. **Deploy Base Configuration**
   ```bash
   kubectl apply -k infrastructure/kubernetes/base
   ```

2. **Deploy Environment-Specific Overlays**
   ```bash
   # Development
   kubectl apply -k infrastructure/kubernetes/overlays/development
   
   # Production
   kubectl apply -k infrastructure/kubernetes/overlays/production
   ```

### Configuration

#### **Production Values (values-production.yaml)**
```yaml
# Production configuration
environment: production
debug: false

# API Gateway
apiGateway:
  replicaCount: 5
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"
  autoscaling:
    enabled: true
    minReplicas: 5
    maxReplicas: 20

# Database
postgresql:
  primary:
    persistence:
      size: 500Gi
    resources:
      requests:
        memory: "4Gi"
        cpu: "2000m"
      limits:
        memory: "8Gi"
        cpu: "4000m"

# Monitoring
monitoring:
  enabled: true
  prometheus:
    server:
      persistentVolume:
        size: 200Gi
```

## ☁️ **Cloud Platform Deployment**

### AWS EKS

1. **Create EKS Cluster**
   ```bash
   eksctl create cluster \
     --name rlaas-cluster \
     --version 1.28 \
     --region us-west-2 \
     --nodegroup-name standard-workers \
     --node-type m5.xlarge \
     --nodes 3 \
     --nodes-min 1 \
     --nodes-max 10 \
     --managed
   ```

2. **Add GPU Node Group (Optional)**
   ```bash
   eksctl create nodegroup \
     --cluster rlaas-cluster \
     --region us-west-2 \
     --name gpu-workers \
     --node-type p3.2xlarge \
     --nodes 1 \
     --nodes-min 0 \
     --nodes-max 5 \
     --node-ami-family AmazonLinux2 \
     --node-labels accelerator=nvidia-tesla-v100
   ```

3. **Install NVIDIA Device Plugin**
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
   ```

4. **Deploy RLaaS**
   ```bash
   helm install rlaas rlaas/rlaas -n rlaas-system -f values-aws.yaml
   ```

### Google GKE

1. **Create GKE Cluster**
   ```bash
   gcloud container clusters create rlaas-cluster \
     --zone us-central1-a \
     --machine-type n1-standard-4 \
     --num-nodes 3 \
     --enable-autoscaling \
     --min-nodes 1 \
     --max-nodes 10 \
     --enable-autorepair \
     --enable-autoupgrade
   ```

2. **Add GPU Node Pool**
   ```bash
   gcloud container node-pools create gpu-pool \
     --cluster rlaas-cluster \
     --zone us-central1-a \
     --machine-type n1-standard-4 \
     --accelerator type=nvidia-tesla-v100,count=1 \
     --num-nodes 1 \
     --enable-autoscaling \
     --min-nodes 0 \
     --max-nodes 5
   ```

3. **Install NVIDIA Drivers**
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
   ```

### Azure AKS

1. **Create AKS Cluster**
   ```bash
   az aks create \
     --resource-group rlaas-rg \
     --name rlaas-cluster \
     --node-count 3 \
     --node-vm-size Standard_D4s_v3 \
     --enable-cluster-autoscaler \
     --min-count 1 \
     --max-count 10 \
     --generate-ssh-keys
   ```

2. **Add GPU Node Pool**
   ```bash
   az aks nodepool add \
     --resource-group rlaas-rg \
     --cluster-name rlaas-cluster \
     --name gpupool \
     --node-count 1 \
     --node-vm-size Standard_NC6s_v3 \
     --enable-cluster-autoscaler \
     --min-count 0 \
     --max-count 5
   ```

## 🔧 **Configuration Management**

### Environment Variables

Key environment variables for configuration:

```bash
# Core Configuration
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key

# Database
DATABASE_URL=postgresql://user:pass@host:5432/rlaas
REDIS_URL=redis://host:6379/0

# ML Services
MLFLOW_TRACKING_URI=http://mlflow:5000
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Optimization
NSGA_POPULATION_SIZE=100
NSGA_GENERATIONS=500

# Security
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Secrets Management

1. **Kubernetes Secrets**
   ```bash
   kubectl create secret generic rlaas-secrets \
     --from-literal=postgres-password=secure-password \
     --from-literal=redis-password=secure-password \
     --from-literal=secret-key=secure-secret-key \
     -n rlaas-system
   ```

2. **External Secrets Operator (Recommended)**
   ```yaml
   apiVersion: external-secrets.io/v1beta1
   kind: SecretStore
   metadata:
     name: aws-secrets-manager
   spec:
     provider:
       aws:
         service: SecretsManager
         region: us-west-2
   ```

## 📊 **Monitoring and Observability**

### Prometheus Metrics

RLaaS exposes metrics at `/metrics` endpoint:

- `rlaas_optimization_requests_total`
- `rlaas_optimization_duration_seconds`
- `rlaas_model_training_duration_seconds`
- `rlaas_inference_requests_total`

### Grafana Dashboards

Pre-configured dashboards available:

1. **RLaaS Overview** - System health and performance
2. **Optimization Metrics** - Multi-objective optimization performance
3. **Model Training** - RL training progress and metrics
4. **Infrastructure** - Kubernetes cluster metrics

### Log Aggregation

Configure log forwarding to your preferred system:

```yaml
# Fluent Bit configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off
        Parsers_File  parsers.conf
    
    [INPUT]
        Name              tail
        Path              /var/log/containers/rlaas*.log
        Parser            docker
        Tag               rlaas.*
    
    [OUTPUT]
        Name  es
        Match *
        Host  elasticsearch.logging.svc.cluster.local
        Port  9200
        Index rlaas-logs
```

## 🔒 **Security Considerations**

### Network Security

1. **Network Policies**
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: rlaas-network-policy
   spec:
     podSelector:
       matchLabels:
         app.kubernetes.io/name: rlaas
     policyTypes:
     - Ingress
     - Egress
     ingress:
     - from:
       - podSelector:
           matchLabels:
             app.kubernetes.io/name: rlaas
   ```

2. **TLS Configuration**
   - Use cert-manager for automatic certificate management
   - Enable TLS for all external communications
   - Use mutual TLS for internal service communication

### RBAC

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: rlaas-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
```

## 🚨 **Troubleshooting**

### Common Issues

1. **Pod Startup Issues**
   ```bash
   kubectl describe pod <pod-name> -n rlaas-system
   kubectl logs <pod-name> -n rlaas-system
   ```

2. **Database Connection Issues**
   ```bash
   kubectl exec -it <api-gateway-pod> -n rlaas-system -- \
     psql postgresql://rlaas:password@rlaas-postgresql:5432/rlaas
   ```

3. **Resource Constraints**
   ```bash
   kubectl top nodes
   kubectl top pods -n rlaas-system
   ```

### Health Checks

```bash
# Check all services
kubectl get all -n rlaas-system

# Check health endpoints
kubectl port-forward svc/rlaas-api-gateway 8000:80 -n rlaas-system
curl http://localhost:8000/health
```

## 📈 **Scaling and Performance**

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rlaas-api-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rlaas-api-gateway
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaling

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: rlaas-optimization-engine-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rlaas-optimization-engine
  updatePolicy:
    updateMode: "Auto"
```

## 🔄 **Backup and Recovery**

### Database Backup

```bash
# Create backup
kubectl exec -it rlaas-postgresql-0 -n rlaas-system -- \
  pg_dump -U rlaas rlaas > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
kubectl exec -i rlaas-postgresql-0 -n rlaas-system -- \
  psql -U rlaas -d rlaas < backup_20240101_120000.sql
```

### Model Storage Backup

```bash
# Backup MinIO data
kubectl exec -it rlaas-minio-0 -n rlaas-system -- \
  mc mirror /data/models s3://backup-bucket/models
```

This deployment guide provides comprehensive instructions for deploying RLaaS in various environments with proper security, monitoring, and scaling configurations.
