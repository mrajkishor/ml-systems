from fastapi import FastAPI

from house_price.logging_config import setup_logging

setup_logging()

app = FastAPI(title="House Price Prediction API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
