from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bertopic import BERTopic
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load the embedding model once at startup
embedding_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
topic_model = BERTopic(embedding_model=embedding_model)

class DocumentInput(BaseModel):
    documents: List[str]

    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    "I love using Python for data science.",
                    "FastAPI makes building APIs easy and fast.",
                    "n8n is great for automation workflows."
                ]
            }
        }

@app.get("/")
def health_check():
    return {"message": "BERTopic API is running."}

@app.post("/topics")
async def extract_topics(data: DocumentInput):
    try:
        if not data.documents or len(data.documents) < 2:
            raise HTTPException(
                status_code=422,
                detail="At least 2 documents are required for topic modeling"
            )

        # Fit and transform the documents
        topics, probs = topic_model.fit_transform(data.documents)

        # Convert numpy arrays to lists for JSON serialization
        topics_list = topics.tolist() if isinstance(topics, np.ndarray) else list(topics)

        # Get topic information
        topic_info = topic_model.get_topic_info()
        topic_info_dict = topic_info.to_dict(orient="records")

        return {
            "topics": topics_list,
            "topic_info": topic_info_dict
        }

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing topics: {str(e)}")
