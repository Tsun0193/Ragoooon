import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

# Import RagoonBot from the correct path. Assuming it's in "core.llm.CustomLLM"
from core.llm.CustomLLM import RagoonBot

load_dotenv('../.env')

app = FastAPI(title="RagoonBot API", description="API for RagoonBot, a custom LLM model.", version="0.1a")

"""
This implementation instantiates a new RagoonBot instance for each request, which is not ideal for performance.
In this version, since we are using API calls to generate completions, we can reuse the same instance of RagoonBot for multiple requests.

TODO: Refactor the code to reuse the same instance of RagoonBot for multiple requests.
TODO: Serve SnowFlake Mistral LLM completions APIs instead for RagoonBot.
"""

class CompletionRequest(BaseModel):
    prompt: str
    history: Optional[List[dict]] = None
    model: Optional[str] = Field("mistral-large2", description="The model name to use.")

class CompletionResponse(BaseModel):
    text: str
    llm_model: str = Field(..., description="The model name used to generate the completion.")
    timestamp: str = Field(datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"), description="The timestamp of the completion.")
    additional_info: dict = {}

@app.post("/complete", response_model=CompletionResponse)
def complete_request(request: CompletionRequest):
    try:
        # Create a new instance of RagoonBot with the specified model
        llm = RagoonBot(model=request.model)
        
        response = llm.complete(prompt=request.prompt, history=request.history)
        return CompletionResponse(text=response.text, llm_model=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_complete")
def stream_complete_request(request: CompletionRequest):
    try:
        # Create a new instance of RagoonBot with the specified model
        llm = RagoonBot(model=request.model)
        
        generator = llm.stream_complete(prompt=request.prompt, history=request.history)
        return {"stream": [resp.delta for resp in generator]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))