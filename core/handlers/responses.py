from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class CompletionResponse(BaseModel):
    text: str
    llm_model: str = Field(..., description="The model name used to generate the completion.")
    timestamp: str = Field(datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"), description="The timestamp of the completion.")
    additional_info: dict = {}

class ChatResponse(BaseModel):
    response: str
    updated_history: List[dict]
    llm_model: str = Field(..., description="The model name used to generate the completion.")
    timestamp: str = Field(datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"), description="The timestamp of the completion.")
    additional_info: dict = {}

class RAGCompleteResponse(BaseModel):
    text: str
    rag_model: str = Field(..., description="The model name used to generate the completion.")
    timestamp: str = Field(datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"), description="The timestamp of the completion.")
    additional_info: dict = {}

class RAGChatResponse(BaseModel):
    response: str
    updated_history: List[dict]
    rag_model: str = Field(..., description="The model name used to generate the completion.")
    timestamp: str = Field(datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"), description="The timestamp of the completion.")
    additional_info: dict = {}