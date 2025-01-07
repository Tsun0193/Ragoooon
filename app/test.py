import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Import RagoonBot from the correct path. Assuming it's in "core.llm.CustomLLM"
from core.llm.CustomLLM import RagoonBot
from core.rag.RAG import Rag
from core.handlers.requests import *
from core.handlers.responses import CompletionResponse, RAGCompleteResponse
from geo.utils import plot_route, get_destination, calculate_route
from asr.whisper import *

load_dotenv('../.env')
# os.chdir("../")

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
    
@app.post("/rag", response_model=RAGCompleteResponse)
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
        response = rag.complete(prompts=request.prompt)


        return RAGCompleteResponse(text=response.text, rag_model=request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_rag")
def rag_stream_complete_request(request: RAGChatRequest):
    """
    Retrieve and generate a streamed response using the RAG model.

    :param request: RAGCompleteRequest containing the user prompt.
    :return: The RAG completion response.
    """
    try:
        # Create a new instance of RagoonBot with the specified model
        llm = RagoonBot(model=request.model)
        rag = Rag(llm=llm)

        # Generate the completion response
        generator = rag.stream_complete(prompts=request.prompts, history=request.history)

        updated_history = request.history if request.history else []
        updated_history.append({"role": "User", "content": request.prompts})

        return {"stream": [resp.delta for resp in generator],
                "updated_history": updated_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/route")
def route_request(request: RouteRequest):
    """
    Calculate and plot the route between two coordinates.

    :param start_coords: The starting coordinates.
    :param end_coords: The destination coordinates.
    :return: The route geometry and distance in kilometers.
    """
    try:
        start_coords = request.current_location
        end_coords = get_destination(destination=request.destination)
        if end_coords:
            route, distance = calculate_route(start_coords, end_coords)
            if route:
                m = plot_route(route, start_coords, end_coords)
                return {"route": route, "distance": distance, "map": m}
            else:
                raise HTTPException(status_code=500, detail="Error fetching route.")
        else:
            raise HTTPException(status_code=500, detail="Could not find the destination address.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/asr")
def asr_request(request: AudioRequest):
    """
    Transcribe the audio file using the Whisper model.

    :param request: AudioRequest containing the audio file.
    :return: The transcribed text.
    """
    try:
        os.chdir("../")
        response = query(request.audio_file)
        os.chdir("app")
        return {"transcription": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
