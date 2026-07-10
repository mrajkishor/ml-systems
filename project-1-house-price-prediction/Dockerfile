FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .

# Bake a trained model into the image so the container is self-contained (no external
# artifact store yet). Fewer Optuna trials here keeps the build fast; the full 30-trial
# search still runs in scripts/run_training.py for local/CI use.
ENV OPTUNA_TRIALS=15
RUN python scripts/run_training.py

EXPOSE 8000
CMD ["uvicorn", "house_price.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
