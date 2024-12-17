import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from together import Together
from huggingface_hub import InferenceClient
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

load_dotenv('../../.env')

def complete(
    user_text: str,
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    history: Optional[List[dict]] = None,
    max_tokens: int = 256
) -> str:
    if history:
        messages = history.copy()  # Avoid mutating the original history
    else:
        messages = []

    messages.append({
        "role": "user",
        "content": user_text
    })

    # Decide which client to use based on the model
    if model == "meta-llama/Llama-3.2-3B-Instruct":
        client = InferenceClient(api_key=os.environ["HF_TOKEN"])
    else:
        client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to generate completion: {e}") from e


class RagoonBot(CustomLLM):
    """
    RagoonBot is a custom LLM model that uses the Meta-LLAMA API to generate text completions.
    
    :param model_name: str, default "mistral-large2". The model name to use.
    :param context_window: int, default 3900. The context window size.
    :param num_output: int, default 256. The number of output tokens.
    """

    llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    context_window: int = 3900
    num_output: int = 256

    def __init__(self, llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct", 
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.llm_model_name = llm_model_name
        print(f"RagoonBot initialized with model: {self.llm_model_name}")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.llm_model_name,
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        response = complete(
            user_text=prompt,
            model=self.llm_model_name,
            history=history,
            max_tokens=self.num_output
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> CompletionResponseGen:
        try:
            full_response = complete(
                user_text=prompt,
                model=self.llm_model_name,
                history=history,
                max_tokens=self.num_output
            )
        except Exception as e:
            yield CompletionResponse(text="", delta=f"Error: {e}")
            return

        accumulated_text = ""
        for char in full_response:
            accumulated_text += char
            yield CompletionResponse(text=accumulated_text, delta=char)


if __name__ == "__main__":
    llm = RagoonBot(model_name="meta-llama/Llama-3.2-3B-Instruct")
    print(llm.complete("Hello, how are you?").text)
