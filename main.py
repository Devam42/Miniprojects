from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class TextInput(BaseModel):
    text: str

catboost_pipeline = joblib.load("catboost_text_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

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
    
    sentiment_str = label_encoder.inverse_transform(numeric_pred)
    
    return {
        "input_text": input_text,
        "predicted_sentiment": sentiment_str[0]
    }
