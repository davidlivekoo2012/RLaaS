# Development startup script for RLaaS platform (PowerShell)

param(
    [string]$Action = "start"
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

# Check if Docker is running
function Test-Docker {
    try {
        docker info | Out-Null
        Write-Success "Docker is running"
        return $true
    }
    catch {
        Write-Error "Docker is not running. Please start Docker Desktop first."
        return $false
    }
}

# Check if docker-compose is available
function Test-DockerCompose {
    try {
        docker-compose --version | Out-Null
        Write-Success "docker-compose is available"
        return $true
    }
    catch {
        Write-Error "docker-compose is not installed. Please install Docker Desktop with docker-compose."
        return $false
    }
}

# Create .env file if it doesn't exist
function Initialize-Environment {
    if (-not (Test-Path ".env")) {
        Write-Info "Creating .env file from template..."
        Copy-Item ".env.example" ".env"
        Write-Success ".env file created. Please review and modify as needed."
    }
    else {
        Write-Info ".env file already exists"
    }
}

# Start the development environment
function Start-Services {
    Write-Info "Starting RLaaS development environment..."
    
    # Pull latest images
    Write-Info "Pulling latest Docker images..."
    docker-compose pull
    
    # Build and start services
    Write-Info "Building and starting services..."
    docker-compose up -d --build
    
    # Wait for services to be ready
    Write-Info "Waiting for services to be ready..."
    Start-Sleep -Seconds 10
    
    # Check service health
    Test-ServicesHealth
}

# Check if services are healthy
function Test-ServicesHealth {
    Write-Info "Checking service health..."
    
    # Check API Gateway
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "API Gateway is healthy"
        }
    }
    catch {
        Write-Warning "API Gateway is not responding yet"
    }
    
    # Check Web Console
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "Web Console is healthy"
        }
    }
    catch {
        Write-Warning "Web Console is not responding yet"
    }
    
    # Check MLflow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "MLflow is healthy"
        }
    }
    catch {
        Write-Warning "MLflow is not responding yet"
    }
}

# Show service URLs
function Show-ServiceUrls {
    Write-Host ""
    Write-Info "RLaaS Development Environment is ready!"
    Write-Host ""
    Write-Host "Service URLs:"
    Write-Host "  ðŸŒ API Gateway:    http://localhost:8000"
    Write-Host "  ðŸ“Š API Docs:       http://localhost:8000/docs"
    Write-Host "  ðŸ–¥ï¸  Web Console:    http://localhost:8080"
    Write-Host "  ðŸ“ˆ MLflow:         http://localhost:5000"
    Write-Host "  ðŸ“Š Grafana:        http://localhost:3000 (admin/admin)"
    Write-Host "  ðŸ” Prometheus:     http://localhost:9090"
    Write-Host ""
    Write-Host "Useful commands:"
    Write-Host "  ðŸ“‹ View logs:      docker-compose logs -f"
    Write-Host "  ðŸ›‘ Stop services:  docker-compose down"
    Write-Host "  ðŸ”„ Restart:        docker-compose restart"
    Write-Host "  ðŸ§¹ Clean up:       docker-compose down -v"
    Write-Host ""
    Write-Host "CLI Usage:"
    Write-Host "  rlaas health"
    Write-Host "  rlaas optimize templates"
    Write-Host "  rlaas optimize start --problem-type 5g --algorithm nsga3"
    Write-Host ""
}

# Main execution
function Start-Main {
    Write-Info "Starting RLaaS development environment setup..."
    
    if (-not (Test-Docker)) {
        exit 1
    }
    
    if (-not (Test-DockerCompose)) {
        exit 1
    }
    
    Initialize-Environment
    Start-Services
    Show-ServiceUrls
    
    Write-Success "Development environment is ready!"
}

# Handle script actions
switch ($Action.ToLower()) {
    "stop" {
        Write-Info "Stopping RLaaS development environment..."
        docker-compose down
        Write-Success "Services stopped"
    }
    "restart" {
        Write-Info "Restarting RLaaS development environment..."
        docker-compose restart
        Write-Success "Services restarted"
    }
    "logs" {
        docker-compose logs -f
    }
    "clean" {
        Write-Info "Cleaning up RLaaS development environment..."
        docker-compose down -v
        docker system prune -f
        Write-Success "Environment cleaned up"
    }
    "status" {
        docker-compose ps
    }
    default {
        Start-Main
    }
}
