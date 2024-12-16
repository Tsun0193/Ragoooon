import os
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata, LLM
from llama_index.core.llms.callbacks import llm_completion_callback
from huggingface_hub import InferenceClient
from typing import Any, List
from dotenv import load_dotenv

load_dotenv('../../.env')

client = InferenceClient(api_key=os.environ["HF_TOKEN"])

def complete(user_text,
             model = "meta-llama/Llama-3.2-3B-Instruct",
             history: List[dict] = None) -> str:
    # completion = Complete(
    #     model="snowflake-arctic",
    #     prompt=user_text,
    #     session=snowflake_session,
    # )
    # return completion

    if history:
        messages = history
    else:
        messages = []

    messages.append({
        "role": "user",
        "content": user_text
    })

    completion = client.chat.completions.create(
        model=model, 
        messages=messages, 
        max_tokens=256
    )

    return completion.choices[0].message.content

class RagoonBot(CustomLLM):
    """
    RagoonBot is a custom LLM model that uses the Meta-LLAMA API to generate text completions.
    
    :param context_window: int, default 3900. The context window size.
    :param num_output: int, default 256. The number of output tokens.
    :param model_name: str, default "mistral-large2". The model name to use.
    """
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "mistral-large2"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = complete(prompt)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        # In streaming mode, we'll still receive the full response at the end of generate.
        # To truly stream token by token, you'd need to yield from within the generate function itself.
        # Here we simulate token-level streaming by splitting the final response.
        full_response = complete(prompt)

        accumulated_text = ""
        for char in full_response:
            accumulated_text += char
            yield CompletionResponse(text=accumulated_text, delta=char)
