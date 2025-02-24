from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pymongo import MongoClient
from datetime import datetime

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Load your model and label encoder
catboost_pipeline = joblib.load("catboost_text_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 1. Put your MongoDB Atlas connection string here,
#    replacing <db_password> with your real password.
MONGO_CONNECTION_STRING = (
    "mongodb+srv://devamkathane:<db_password>@cluster0.k0s0e.mongodb.net/"
    "?retryWrites=true&w=majority&appName=Cluster0"
)

# 2. Create the client, specify database and collection
client = MongoClient(MONGO_CONNECTION_STRING)
db = client["sample_mflix"]           # <--- Database name from your screenshot
onelab_collection = db["Onelab Task2"] # <--- Collection name

@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Hello from the Text Classifier API!"}

@app.post("/predict")
def predict_sentiment(payload: TextInput):
    """
    POST endpoint that accepts text and returns the predicted sentiment.
    Also logs the request+prediction to MongoDB.
    """
    input_text = payload.text

    # Predict numeric label and convert to string label
    numeric_pred = catboost_pipeline.predict([input_text])
    sentiment_str = label_encoder.inverse_transform(numeric_pred)[0]

    # 3. Insert a log document into the 'Onelab Task2' collection
    log_doc = {
        "timestamp": datetime.utcnow(),
        "input_text": input_text,
        "predicted_sentiment": sentiment_str
    }
    onelab_collection.insert_one(log_doc)

    return {
        "input_text": input_text,
        "predicted_sentiment": sentiment_str
    }
