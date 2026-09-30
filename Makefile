.PHONY: help up down restart logs ps test lint clean

# Default shell
SHELL := /bin/bash

help: ## Show available make commands
	@echo "Available commands in Enterprise AI Platform:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Start all local services via Docker Compose
	docker compose up -d --build

down: ## Stop all local services
	docker compose down --remove-orphans

restart: down up ## Restart all services

logs: ## View container logs in real time
	docker compose logs -f

ps: ## List status of all platform containers
	docker compose ps

test: test-gateway test-rag test-mlops ## Run all unit & integration tests across services

test-gateway: ## Run tests for API Gateway (Go)
	@echo "Running API Gateway tests..."
	@cd services/api-gateway && go test -v ./... || true

test-rag: ## Run tests for Hybrid RAG Engine (Python)
	@echo "Running Hybrid RAG tests..."
	@cd services/hybrid-rag-engine && pytest || true

test-mlops: ## Run tests for MLOps Inference (Python)
	@echo "Running MLOps Inference tests..."
	@cd services/mlops-inference && pytest || true

lint: ## Run linters across Go and Python codebases
	@echo "Linting API Gateway (golangci-lint)..."
	@cd services/api-gateway && golangci-lint run || true
	@echo "Linting Python microservices (ruff)..."
	@ruff check services/ || true

clean: ## Clean up temporary files, caches, and test artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
