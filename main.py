from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from typing import List

app = FastAPI()

class Documents(BaseModel):
    docs: List[str]

# Load model once at startup
topic_model = BERTopic()

@app.get("/")
def health_check():
    return {"message": "BERTopic API is running."}

@app.post("/topics")
def extract_topics(data: Documents):
    topics, _ = topic_model.fit_transform(data.docs)
    return {"topics": topics}
