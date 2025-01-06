from pydantic import BaseModel, Field
from typing import Optional, List, Tuple

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
    prompts: str
    history: Optional[List[dict]] = Field([], description="The history of previous interactions.")
    model: Optional[str] = Field("mistral-large2", description="The model name to use.")

class RouteRequest(BaseModel):
    destination: str
    current_location: Tuple[float, float] = Field(None, description="The current location coordinates.")