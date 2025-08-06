# RLaaS Platform Makefile

.PHONY: help install dev test lint format clean build deploy docs

# Default target
help:
	@echo "RLaaS Platform Development Commands"
	@echo "=================================="
	@echo "install     Install dependencies"
	@echo "dev         Start development environment"
	@echo "test        Run tests"
	@echo "lint        Run linting"
	@echo "format      Format code"
	@echo "clean       Clean build artifacts"
	@echo "build       Build Docker images"
	@echo "deploy      Deploy to Kubernetes"
	@echo "docs        Build documentation"
	@echo "setup       Initial project setup"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Start development environment
dev:
	docker-compose up -d
	@echo "Development environment started!"
	@echo "API Gateway: http://localhost:8000"
	@echo "Web Console: http://localhost:8080"
	@echo "MLflow: http://localhost:5000"
	@echo "Grafana: http://localhost:3000"

# Stop development environment
dev-stop:
	docker-compose down

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Build Docker images
build:
	docker build -t rlaas:latest .
	docker build -t rlaas:dev --target development .

# Deploy to Kubernetes
deploy:
	./scripts/deploy.sh

# Deploy to local development
deploy-local:
	./scripts/deploy.sh --environment local

# Build documentation
docs:
	mkdocs build

# Serve documentation locally
docs-serve:
	mkdocs serve

# Initial project setup
setup:
	cp .env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run 'make install' to install dependencies"

# Database migrations
migrate:
	alembic upgrade head

# Create new migration
migration:
	alembic revision --autogenerate -m "$(MSG)"

# Reset database
reset-db:
	docker-compose exec postgres psql -U rlaas -d rlaas -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	make migrate

# Run security checks
security:
	bandit -r src/
	safety check

# Run performance tests
perf-test:
	locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Generate API documentation
api-docs:
	python -c "import uvicorn; uvicorn.run('rlaas.api.main:app', host='0.0.0.0', port=8000)" &
	sleep 5
	curl http://localhost:8000/openapi.json > docs/api/openapi.json
	pkill -f uvicorn

# Backup data
backup:
	kubectl exec -n rlaas-system deployment/postgresql -- pg_dump -U rlaas rlaas > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore data
restore:
	kubectl exec -i -n rlaas-system deployment/postgresql -- psql -U rlaas -d rlaas < $(FILE)

# Monitor logs
logs:
	kubectl logs -f -n rlaas-system -l app.kubernetes.io/name=rlaas

# Port forward services
port-forward:
	kubectl port-forward -n rlaas-system svc/rlaas-api-gateway 8000:80 &
	kubectl port-forward -n rlaas-system svc/rlaas-web-console 8080:80 &
	kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80 &
	@echo "Services forwarded:"
	@echo "API Gateway: http://localhost:8000"
	@echo "Web Console: http://localhost:8080"
	@echo "Grafana: http://localhost:3000"

# Check code quality
quality: lint test security
	@echo "Code quality checks completed!"

# Full CI pipeline
ci: clean install quality build
	@echo "CI pipeline completed successfully!"

# Release preparation
release:
	@echo "Preparing release..."
	python scripts/bump_version.py
	git add .
	git commit -m "Bump version for release"
	git tag v$(shell python -c "import src.rlaas; print(src.rlaas.__version__)")
	@echo "Release prepared. Push with: git push origin main --tags"
