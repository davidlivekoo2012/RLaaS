#!/bin/bash

# Development startup script for RLaaS platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    log_success "Docker is running"
}

# Check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed. Please install docker-compose first."
        exit 1
    fi
    log_success "docker-compose is available"
}

# Create .env file if it doesn't exist
setup_env() {
    if [ ! -f .env ]; then
        log_info "Creating .env file from template..."
        cp .env.example .env
        log_success ".env file created. Please review and modify as needed."
    else
        log_info ".env file already exists"
    fi
}

# Start the development environment
start_services() {
    log_info "Starting RLaaS development environment..."
    
    # Pull latest images
    log_info "Pulling latest Docker images..."
    docker-compose pull
    
    # Build and start services
    log_info "Building and starting services..."
    docker-compose up -d --build
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_services_health
}

# Check if services are healthy
check_services_health() {
    log_info "Checking service health..."
    
    # Check API Gateway
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API Gateway is healthy"
    else
        log_warning "API Gateway is not responding yet"
    fi
    
    # Check Web Console
    if curl -f http://localhost:8080 > /dev/null 2>&1; then
        log_success "Web Console is healthy"
    else
        log_warning "Web Console is not responding yet"
    fi
    
    # Check MLflow
    if curl -f http://localhost:5000 > /dev/null 2>&1; then
        log_success "MLflow is healthy"
    else
        log_warning "MLflow is not responding yet"
    fi
}

# Show service URLs
show_urls() {
    echo ""
    log_info "RLaaS Development Environment is ready!"
    echo ""
    echo "Service URLs:"
    echo "  🌐 API Gateway:    http://localhost:8000"
    echo "  📊 API Docs:       http://localhost:8000/docs"
    echo "  🖥️  Web Console:    http://localhost:8080"
    echo "  📈 MLflow:         http://localhost:5000"
    echo "  📊 Grafana:        http://localhost:3000 (admin/admin)"
    echo "  🔍 Prometheus:     http://localhost:9090"
    echo ""
    echo "Useful commands:"
    echo "  📋 View logs:      docker-compose logs -f"
    echo "  🛑 Stop services:  docker-compose down"
    echo "  🔄 Restart:        docker-compose restart"
    echo "  🧹 Clean up:       docker-compose down -v"
    echo ""
    echo "CLI Usage:"
    echo "  rlaas health"
    echo "  rlaas optimize templates"
    echo "  rlaas optimize start --problem-type 5g --algorithm nsga3"
    echo ""
}

# Main execution
main() {
    log_info "Starting RLaaS development environment setup..."
    
    check_docker
    check_docker_compose
    setup_env
    start_services
    show_urls
    
    log_success "Development environment is ready!"
}

# Handle script arguments
case "${1:-}" in
    "stop")
        log_info "Stopping RLaaS development environment..."
        docker-compose down
        log_success "Services stopped"
        ;;
    "restart")
        log_info "Restarting RLaaS development environment..."
        docker-compose restart
        log_success "Services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "clean")
        log_info "Cleaning up RLaaS development environment..."
        docker-compose down -v
        docker system prune -f
        log_success "Environment cleaned up"
        ;;
    "status")
        docker-compose ps
        ;;
    *)
        main
        ;;
esac
