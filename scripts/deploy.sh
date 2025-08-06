#!/bin/bash

# RLaaS Platform Deployment Script
# This script deploys the RLaaS platform to a Kubernetes cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-rlaas-system}
ENVIRONMENT=${ENVIRONMENT:-development}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-localhost:5000}
HELM_RELEASE_NAME=${HELM_RELEASE_NAME:-rlaas}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed. Please install helm first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed. Please install docker first."
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    log_success "All prerequisites are met."
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists."
    else
        kubectl create namespace $NAMESPACE
        log_success "Namespace $NAMESPACE created."
    fi
    
    # Label the namespace
    kubectl label namespace $NAMESPACE app.kubernetes.io/name=rlaas --overwrite
    kubectl label namespace $NAMESPACE app.kubernetes.io/instance=$HELM_RELEASE_NAME --overwrite
}

build_images() {
    log_info "Building Docker images..."
    
    # Build main application image
    docker build -t $DOCKER_REGISTRY/rlaas:latest .
    
    # Build web console image
    if [ -d "src/ui/web-console" ]; then
        docker build -t $DOCKER_REGISTRY/rlaas-web-console:latest src/ui/web-console/
    fi
    
    log_success "Docker images built successfully."
}

push_images() {
    log_info "Pushing Docker images to registry..."
    
    docker push $DOCKER_REGISTRY/rlaas:latest
    
    if [ -d "src/ui/web-console" ]; then
        docker push $DOCKER_REGISTRY/rlaas-web-console:latest
    fi
    
    log_success "Docker images pushed successfully."
}

deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Deploy PostgreSQL
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install postgresql bitnami/postgresql \
        --namespace $NAMESPACE \
        --set auth.postgresPassword=rlaas \
        --set auth.database=rlaas \
        --set primary.persistence.size=10Gi \
        --wait
    
    # Deploy Redis
    helm upgrade --install redis bitnami/redis \
        --namespace $NAMESPACE \
        --set auth.enabled=false \
        --set master.persistence.size=5Gi \
        --wait
    
    # Deploy Kafka
    helm upgrade --install kafka bitnami/kafka \
        --namespace $NAMESPACE \
        --set persistence.size=10Gi \
        --wait
    
    log_success "Infrastructure components deployed."
}

deploy_application() {
    log_info "Deploying RLaaS application..."
    
    # Deploy using Helm chart
    helm upgrade --install $HELM_RELEASE_NAME infrastructure/helm/rlaas \
        --namespace $NAMESPACE \
        --set image.repository=$DOCKER_REGISTRY/rlaas \
        --set image.tag=latest \
        --set environment=$ENVIRONMENT \
        --wait
    
    log_success "RLaaS application deployed."
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=rlaas -n $NAMESPACE --timeout=300s
    
    # Check service status
    kubectl get pods,services,ingress -n $NAMESPACE
    
    log_success "Deployment verification completed."
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Deploy Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set grafana.adminPassword=admin \
        --wait
    
    log_success "Monitoring setup completed."
}

main() {
    log_info "Starting RLaaS platform deployment..."
    
    check_prerequisites
    create_namespace
    
    if [ "$ENVIRONMENT" != "local" ]; then
        build_images
        push_images
    fi
    
    deploy_infrastructure
    deploy_application
    verify_deployment
    
    if [ "$ENVIRONMENT" = "production" ]; then
        setup_monitoring
    fi
    
    log_success "RLaaS platform deployment completed successfully!"
    
    # Display access information
    echo ""
    log_info "Access Information:"
    echo "  Web Console: kubectl port-forward -n $NAMESPACE svc/rlaas-web-console 8080:80"
    echo "  API Gateway: kubectl port-forward -n $NAMESPACE svc/rlaas-api-gateway 8000:80"
    echo "  Grafana: kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    echo ""
    log_info "Use 'kubectl get all -n $NAMESPACE' to check the status of all resources."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --namespace NAMESPACE    Kubernetes namespace (default: rlaas-system)"
            echo "  --environment ENV        Environment (development/staging/production)"
            echo "  --registry REGISTRY      Docker registry (default: localhost:5000)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
