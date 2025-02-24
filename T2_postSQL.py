from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import psycopg2
from datetime import datetime

app = FastAPI()

catboost_pipeline = joblib.load("catboost_text_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class TextInput(BaseModel):
    text: str

POSTGRES_DB = "Onelab Task2"     
POSTGRES_USER = "postgres"     
POSTGRES_PASSWORD = "Devam@42" 
POSTGRES_HOST = "localhost"    
POSTGRES_PORT = 5432


conn = psycopg2.connect(
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT
)
conn.autocommit = True


create_table_query = """
CREATE TABLE IF NOT EXISTS predictions_log (
    id SERIAL PRIMARY KEY,
    input_text TEXT NOT NULL,
    predicted_sentiment TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""
with conn.cursor() as cur:
    cur.execute(create_table_query)


@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {"message": "Welcome to the Text Classifier API + PostgreSQL!"}

@app.post("/predict")
def predict_sentiment(payload: TextInput):
    """
    POST endpoint that accepts text and returns the predicted sentiment,
    then logs the request in the predictions_log table.
    """
    input_text = payload.text

    numeric_pred = catboost_pipeline.predict([input_text])
    sentiment_str = label_encoder.inverse_transform(numeric_pred)[0]


    insert_query = """
        INSERT INTO predictions_log (input_text, predicted_sentiment)
        VALUES (%s, %s)
    """
    with conn.cursor() as cur:
        cur.execute(insert_query, (input_text, sentiment_str))

    return {
        "input_text": input_text,
        "predicted_sentiment": sentiment_str
    }
