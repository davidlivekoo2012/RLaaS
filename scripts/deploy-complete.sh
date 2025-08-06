#!/bin/bash

# RLaaS Complete Deployment Script
# This script deploys the complete RLaaS platform to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="rlaas-system"
DOCKER_REGISTRY="rlaas"
VERSION="latest"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for deployment
wait_for_deployment() {
    local deployment=$1
    local namespace=$2
    local timeout=${3:-300}
    
    print_status "Waiting for deployment $deployment to be ready..."
    
    if kubectl wait --for=condition=available --timeout=${timeout}s deployment/$deployment -n $namespace; then
        print_success "Deployment $deployment is ready"
        return 0
    else
        print_error "Deployment $deployment failed to become ready within ${timeout}s"
        return 1
    fi
}

# Function to wait for pod
wait_for_pod() {
    local label=$1
    local namespace=$2
    local timeout=${3:-300}
    
    print_status "Waiting for pod with label $label to be ready..."
    
    if kubectl wait --for=condition=ready --timeout=${timeout}s pod -l $label -n $namespace; then
        print_success "Pod with label $label is ready"
        return 0
    else
        print_error "Pod with label $label failed to become ready within ${timeout}s"
        return 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists kubectl; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command_exists docker; then
        print_error "docker is not installed"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info >/dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build API image
    print_status "Building RLaaS API image..."
    docker build -t ${DOCKER_REGISTRY}/api:${VERSION} -f docker/Dockerfile.api .
    
    # Build Worker image
    print_status "Building RLaaS Worker image..."
    docker build -t ${DOCKER_REGISTRY}/worker:${VERSION} -f docker/Dockerfile.worker .
    
    # Build Training Worker image
    print_status "Building RLaaS Training Worker image..."
    docker build -t ${DOCKER_REGISTRY}/training-worker:${VERSION} -f docker/Dockerfile.training-worker .
    
    # Build Inference image
    print_status "Building RLaaS Inference image..."
    docker build -t ${DOCKER_REGISTRY}/inference:${VERSION} -f docker/Dockerfile.inference .
    
    print_success "Docker images built successfully"
}

# Deploy infrastructure components
deploy_infrastructure() {
    print_status "Deploying infrastructure components..."
    
    # Apply complete deployment
    kubectl apply -f infrastructure/k8s/complete-deployment.yaml
    
    # Wait for database to be ready
    wait_for_deployment "postgres" $NAMESPACE 300
    
    # Wait for Redis to be ready
    wait_for_deployment "redis" $NAMESPACE 180
    
    # Wait for Zookeeper to be ready
    wait_for_deployment "zookeeper" $NAMESPACE 180
    
    # Wait for Kafka to be ready
    wait_for_deployment "kafka" $NAMESPACE 300
    
    # Wait for MinIO to be ready
    wait_for_deployment "minio" $NAMESPACE 180
    
    # Wait for MLflow to be ready
    wait_for_deployment "mlflow" $NAMESPACE 300
    
    print_success "Infrastructure components deployed successfully"
}

# Deploy RLaaS services
deploy_services() {
    print_status "Deploying RLaaS services..."
    
    # Wait for API to be ready
    wait_for_deployment "rlaas-api" $NAMESPACE 300
    
    # Wait for Workers to be ready
    wait_for_deployment "rlaas-worker" $NAMESPACE 300
    wait_for_deployment "rlaas-training-worker" $NAMESPACE 300
    
    # Wait for Inference service to be ready
    wait_for_deployment "rlaas-inference" $NAMESPACE 300
    
    print_success "RLaaS services deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    print_status "Deploying monitoring components..."
    
    # Apply monitoring configuration
    kubectl apply -f infrastructure/k8s/ingress-monitoring.yaml
    
    # Wait for Prometheus to be ready
    wait_for_deployment "prometheus" $NAMESPACE 300
    
    # Wait for Grafana to be ready
    wait_for_deployment "grafana" $NAMESPACE 300
    
    print_success "Monitoring components deployed successfully"
}

# Initialize database
initialize_database() {
    print_status "Initializing database..."
    
    # Wait for database pod to be ready
    wait_for_pod "app=postgres" $NAMESPACE 300
    
    # Run database migrations
    kubectl exec -n $NAMESPACE deployment/rlaas-api -- python -m alembic upgrade head
    
    print_success "Database initialized successfully"
}

# Create MinIO buckets
create_minio_buckets() {
    print_status "Creating MinIO buckets..."
    
    # Wait for MinIO to be ready
    wait_for_pod "app=minio" $NAMESPACE 300
    
    # Create buckets using MinIO client
    kubectl exec -n $NAMESPACE deployment/minio -- mc alias set local http://localhost:9000 minioadmin minioadmin
    kubectl exec -n $NAMESPACE deployment/minio -- mc mb local/mlflow-artifacts --ignore-existing
    kubectl exec -n $NAMESPACE deployment/minio -- mc mb local/rlaas-models --ignore-existing
    kubectl exec -n $NAMESPACE deployment/minio -- mc mb local/rlaas-data --ignore-existing
    
    print_success "MinIO buckets created successfully"
}

# Verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check all pods are running
    print_status "Checking pod status..."
    kubectl get pods -n $NAMESPACE
    
    # Check services
    print_status "Checking services..."
    kubectl get services -n $NAMESPACE
    
    # Check ingress
    print_status "Checking ingress..."
    kubectl get ingress -n $NAMESPACE
    
    # Test API health
    print_status "Testing API health..."
    API_POD=$(kubectl get pods -n $NAMESPACE -l app=rlaas-api -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec -n $NAMESPACE $API_POD -- curl -f http://localhost:8000/health; then
        print_success "API health check passed"
    else
        print_warning "API health check failed"
    fi
    
    print_success "Deployment verification completed"
}

# Display access information
show_access_info() {
    print_status "Deployment completed successfully!"
    echo
    echo "=== Access Information ==="
    echo
    echo "To access the services, you can use port-forwarding:"
    echo
    echo "RLaaS API:"
    echo "  kubectl port-forward -n $NAMESPACE svc/rlaas-api 8000:8000"
    echo "  Then access: http://localhost:8000"
    echo
    echo "RLaaS Inference:"
    echo "  kubectl port-forward -n $NAMESPACE svc/rlaas-inference 8001:8001"
    echo "  Then access: http://localhost:8001"
    echo
    echo "MLflow:"
    echo "  kubectl port-forward -n $NAMESPACE svc/mlflow 5000:5000"
    echo "  Then access: http://localhost:5000"
    echo
    echo "Grafana:"
    echo "  kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000"
    echo "  Then access: http://localhost:3000 (admin/admin123)"
    echo
    echo "Prometheus:"
    echo "  kubectl port-forward -n $NAMESPACE svc/prometheus 9090:9090"
    echo "  Then access: http://localhost:9090"
    echo
    echo "MinIO Console:"
    echo "  kubectl port-forward -n $NAMESPACE svc/minio 9001:9001"
    echo "  Then access: http://localhost:9001 (minioadmin/minioadmin)"
    echo
    echo "=== Web Console ==="
    echo "To start the web console:"
    echo "  streamlit run src/rlaas/ui/console/app.py"
    echo
    echo "=== SDK Usage ==="
    echo "To use the Python SDK:"
    echo "  from rlaas.sdk import RLaaSClient"
    echo "  client = RLaaSClient('http://localhost:8000')"
    echo
}

# Cleanup function
cleanup() {
    print_status "Cleaning up deployment..."
    
    kubectl delete -f infrastructure/k8s/ingress-monitoring.yaml --ignore-not-found=true
    kubectl delete -f infrastructure/k8s/complete-deployment.yaml --ignore-not-found=true
    
    print_success "Cleanup completed"
}

# Main deployment function
main() {
    echo "🚀 RLaaS Complete Deployment Script"
    echo "===================================="
    echo
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            build_images
            deploy_infrastructure
            sleep 30  # Wait for infrastructure to stabilize
            deploy_services
            deploy_monitoring
            sleep 10  # Wait for services to stabilize
            initialize_database
            create_minio_buckets
            verify_deployment
            show_access_info
            ;;
        "cleanup")
            cleanup
            ;;
        "verify")
            verify_deployment
            ;;
        "info")
            show_access_info
            ;;
        *)
            echo "Usage: $0 [deploy|cleanup|verify|info]"
            echo
            echo "Commands:"
            echo "  deploy   - Deploy complete RLaaS platform (default)"
            echo "  cleanup  - Remove RLaaS deployment"
            echo "  verify   - Verify deployment status"
            echo "  info     - Show access information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
