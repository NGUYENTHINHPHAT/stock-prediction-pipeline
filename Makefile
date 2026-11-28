.PHONY: help install train validate tune backtest serve test docker-up docker-down clean

help:
	@echo "make install    Install dependencies and create .env"
	@echo "make train      Train the model -> models/saved/latest_model.pkl"
	@echo "make validate   Purged walk-forward cross-validation"
	@echo "make tune       Optuna hyperparameter search -> models/best_params.json"
	@echo "make backtest   Backtest vs baselines + significance tests -> results/"
	@echo "make serve      Run the API + dashboard at http://localhost:8000/dashboard"
	@echo "make test       Run the test suite"
	@echo "make docker-up  Build and run the container"

install:
	pip install -r requirements.txt
	@[ -f .env ] || cp .env.example .env

train:
	python -m scripts.train

validate:
	python -m scripts.validate

tune:
	python -m scripts.tune

backtest:
	python -m scripts.backtest

serve:
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest

docker-up:
	docker compose up --build -d
	@echo "Dashboard: http://localhost:8000/dashboard"

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage
