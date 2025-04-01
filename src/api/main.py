from fastapi import FastAPI
from src.api.routes import predict

app = FastAPI()

app.include_router(predict.router)


@app.get("/")
def root():
    return {"message": "System Anomaly Detection API"}
