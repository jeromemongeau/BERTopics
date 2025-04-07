from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from typing import List

app = FastAPI()

class Documents(BaseModel):
    documents: List[str]  # Changed from 'docs' to 'documents' to match your request format

# Load model once at startup
topic_model = BERTopic()

@app.get("/")
def health_check():
    return {"message": "BERTopic API is running."}

@app.post("/topics")
def extract_topics(data: Documents):
    # Fix syntax error in unpacking and properly handle BERTopic result
    topics, probs = topic_model.fit_transform(data.documents)
    # Return both topics and probabilities
    topic_info = topic_model.get_topic_info()
    
    return {
        "topics": topics.tolist() if hasattr(topics, "tolist") else topics,
        "topic_info": topic_info.to_dict(orient="records") if hasattr(topic_info, "to_dict") else []
    }
