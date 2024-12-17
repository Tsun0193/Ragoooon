import os
from dotenv import load_dotenv
from core.llm.CustomLLM import RagoonBot
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

load_dotenv('../.env')

app = FastAPI(title="RagoonBot API", description="API for RagoonBot, a custom LLM model.", version="0.1")

llm = RagoonBot()

class CompletionRequest(BaseModel):
    prompt: str
    history: Optional[List[dict]] = None

class CompletionResponse(BaseModel):
    text: str
    llm_model: str = Field(llm.llm_model_name, description="The model name used to generate the completion.")
    timestamp: str = Field(datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"), description="The timestamp of the completion.")
    additional_info: dict = {}

@app.post("/complete", response_model=CompletionResponse)
def complete(request: CompletionRequest):
    try:
        response = llm.complete(prompt=request.prompt, history=request.history)
        return CompletionResponse(text=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_complete")
def stream_complete(request: CompletionRequest):
    try:
        generator = llm.stream_complete(prompt=request.prompt, history=request.history)
        return {"stream": [resp.delta for resp in generator]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))