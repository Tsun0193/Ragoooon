import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Import RagoonBot from the correct path. Assuming it's in "core.llm.CustomLLM"
from core.llm.CustomLLM import RagoonBot
from core.rag.RAG import Rag
from core.handlers.requests import CompletionRequest, ChatRequest, RAGCompleteRequest, RAGChatRequest
from core.handlers.responses import CompletionResponse, ChatResponse, RAGCompleteResponse, RAGChatResponse

load_dotenv('../.env')

app = FastAPI(title="RagoonBot API", description="API for RagoonBot, a custom LLM model.", version="0.1a")

@app.post("/complete", response_model=CompletionResponse)
def complete_request(request: CompletionRequest):
    try:
        # Create a new instance of RagoonBot with the specified model
        llm = RagoonBot(model=request.model)
        
        response = llm.complete(prompt=request.prompt)
        return CompletionResponse(text=response.text, llm_model=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_complete")
def stream_complete_request(request: CompletionRequest):
    try:
        # Create a new instance of RagoonBot with the specified model
        llm = RagoonBot(model=request.model)
        
        generator = llm.stream_complete(prompt=request.prompt)
        return {"stream": [resp.delta for resp in generator]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_request(request: ChatRequest):
    """
    Interactive chat endpoint for RagoonBot.

    :param request: CompletionRequest containing the user prompt and history.
    :return: The chat response with updated history.
    """
    try:
        # Create a new instance of RagoonBot with the specified model
        llm = RagoonBot(model=request.model)

        # Generate the completion response
        response = llm.complete(prompt=request.prompt, history=request.history)

        # Append the current interaction to the history
        updated_history = request.history if request.history else []
        updated_history.append({"role": "User", "content": request.prompt})
        updated_history.append({"role": "Assistant", "content": response.text})

        return {
            "response": response.text,
            "updated_history": updated_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_chat")
def stream_chat_request(request: ChatRequest):
    """
    Interactive chat endpoint for RagoonBot with streaming responses.

    :param request: ChatRequest containing the user prompt and history.
    :return: The chat response with updated history.
    """
    try:
        # Create a new instance of RagoonBot with the specified model
        llm = RagoonBot(model=request.model)

        # Generate the completion response
        generator = llm.stream_complete(prompt=request.prompt, history=request.history)

        # Append the current interaction to the history
        updated_history = request.history if request.history else []
        updated_history.append({"role": "User", "content": request.prompt})

        return {
            "stream": [resp.delta for resp in generator],
            "updated_history": updated_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/rag_complete", response_model=RAGCompleteResponse)
def rag_complete_request(request: RAGCompleteRequest):
    """
    Retrieve and generate a response using the RAG model.

    :param request: RAGCompleteRequest containing the user prompt.
    :return: The RAG completion response.
    """
    try:
        # Create a new instance of RagoonBot with the specified model
        llm = RagoonBot(model=request.model)
        rag = Rag(llm=llm)

        # Generate the completion response
        completion_response = rag.complete(prompt=request.prompt)

        # Extract text if the response is of type CompletionResponse
        if isinstance(completion_response, CompletionResponse):
            text = completion_response.text
        else:
            text = completion_response  # Assume it's already a string

        return RAGCompleteResponse(
            text=text,
            rag_model=request.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
