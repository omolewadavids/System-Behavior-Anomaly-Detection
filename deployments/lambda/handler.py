import json
from src.model.inference import load_model, predict_anomaly

model = load_model()

def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        data = body["data"]
        prediction = predict_anomaly(model, data)
        return {
            "statusCode": 200,
            "body": json.dumps({"anomaly_score": prediction})
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
