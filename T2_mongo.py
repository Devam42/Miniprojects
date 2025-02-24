from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
import joblib
from urllib.parse import quote_plus

app = FastAPI()

class TextInput(BaseModel):
    text:str
    
catboost_pipeline = joblib.load("catboost_text_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

USERNAME = "devamkathane"
PASSWORD = "Devam@4231" 


encoded_username = quote_plus(USERNAME)
encoded_password = quote_plus(PASSWORD)


MONGO_CONNECTION_STRING = (
    f"mongodb+srv://{encoded_username}:{encoded_password}"
    "@cluster0.k0s0e.mongodb.net/?retryWrites=true&w=majority"
)

client = MongoClient(MONGO_CONNECTION_STRING)
db=client["sample_mflix"]
onelab_collection = db["Onelab Task2"]

@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {"message": "Welcome to the Text Classifier API!"}


@app.post("/predict")
def predict_sentiment(payload: TextInput):
    """
    POST endpoint that accepts text and returns the predicted sentiment.
    """
    input_text = payload.text

    numeric_pred = catboost_pipeline.predict([input_text])
    
    sentiment_str = label_encoder.inverse_transform(numeric_pred)[0]
    
    log_doc = {
        "timestamp": datetime.utcnow(),
        "input_text": input_text,
        "predicted_sentiment": sentiment_str        
    }
    onelab_collection.insert_one(log_doc)
    
    return {
        "input_text": input_text,
        "predicted_sentiment": sentiment_str[0]
    }
