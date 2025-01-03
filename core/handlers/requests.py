from pydantic import BaseModel, Field
from typing import Optional, List

class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = Field("mistral-large2", description="The model name to use.")

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[dict]] = Field([], description="The history of previous interactions.")
    model: Optional[str] = Field("mistral-large2", description="The model name to use.")

class RAGCompleteRequest(BaseModel):
    prompt: str
    model: Optional[str] = Field("mistral-large2", description="The model name to use.")

class RAGChatRequest(BaseModel):
    prompt: str
    history: Optional[List[dict]] = Field([], description="The history of previous interactions.")
    model: Optional[str] = Field("mistral-large2", description="The model name to use.")