from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bertopic import BERTopic
from typing import List, Optional
import numpy as np

app = FastAPI()

# Update the model to match your exact JSON structure
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
        # Initialize model with each request to ensure fresh state
        # Only do this if you're not doing large-scale processing
        topic_model = BERTopic()
        
        # Make sure data.documents is not empty
        if not data.documents or len(data.documents) < 2:
            raise HTTPException(status_code=422, 
                               detail="At least 2 documents are required for topic modeling")
        
        # Fit and transform the documents
        topics, probs = topic_model.fit_transform(data.documents)
        
        # Convert numpy arrays to lists for JSON serialization
        topics_list = topics.tolist() if isinstance(topics, np.ndarray) else list(topics)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        topic_info_dict = topic_info.to_dict(orient="records")
        
        # Return the results
        return {
            "topics": topics_list,
            "topic_info": topic_info_dict
        }
    except Exception as e:
        # Log the error details
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing topics: {str(e)}")
