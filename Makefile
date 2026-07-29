# Project Bard - Makefile
# Convenience targets for common project operations

.PHONY: help install pipeline train serve chat test clean

help: ## Show this help message
	@echo "Project Bard - Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt

pipeline: ## Run the full data pipeline (download, clean, tokenize, split)
	python data_pipeline.py
	python tokenizer.py
	python dataset.py

train: ## Train the model
	python train.py

evaluate: ## Run evaluation and alignment
	python evaluate.py

serve: ## Start the API server with web UI
	python api_server.py

chat: ## Start interactive CLI chat
	python chat.py

generate: ## Generate sample text
	python generate.py --prompt "To be, or not to be" --stream --max-tokens 150

test: ## Run all tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=. --cov-report=html --tb=short

explore: ## Explore the dataset
	python explore_data.py

voxel-train: ## Train the voxel model
	python voxel_train.py

voxel-chat: ## Chat with the voxel model
	python voxel_chat.py

clean: ## Remove generated data and caches
	rm -rf __pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov .coverage

clean-all: clean ## Remove all generated files (data, checkpoints, etc.)
	rm -rf data/raw/ data/clean/ data/splits/ data/tokenizer/
	rm -rf checkpoints/
	rm -rf wandb/ logs/
	rm -f results_*.jsonl
