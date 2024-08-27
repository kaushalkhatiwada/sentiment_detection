from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load pre-trained model for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

class SentimentRequest(BaseModel):
    text: str

@app.post("/detect")
async def analyze_sentiment(request: SentimentRequest):
    result = sentiment_pipeline(request.text)
    return result

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}
