.PHONY: format lint tb mlflow clean

format:
	uv run ruff format notebooks/ src/
	uv run ruff check --fix notebooks/ src/

lint:
	uv run ruff check notebooks/ src/

tb:
	uv run tensorboard --logdir modellogs

mlflow:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
