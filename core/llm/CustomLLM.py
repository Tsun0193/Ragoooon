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
    platform: str = "huggingface",
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

    # Decide which client to use based on the platform
    if platform == "huggingface":
        client = InferenceClient(api_key=os.environ["HF_TOKEN"])
    elif platform == "together":
        client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    else:
        raise ValueError(f"Unsupported platform: {platform}")

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
    RagoonBot is a custom LLM model that uses the specified platform's API to generate text completions.

    :param model: str, default "meta-llama/Llama-3.2-3B-Instruct". The model name to use.
    :param platform: str, default "huggingface". The platform to use ("huggingface" or "together").
    :param context_window: int, default 3900. The context window size.
    :param num_output: int, default 256. The number of output tokens.
    """

    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    platform: str = "huggingface"
    context_window: int = 3900
    num_output: int = 256

    def __init__(
        self, 
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        platform: str = "huggingface",
        context_window: int = 3900,
        num_output: int = 256,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.model = model
        self.platform = platform
        self.context_window = context_window
        self.num_output = num_output
        print(f"RagoonBot initialized with model: {self.model} and platform: {self.platform}")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
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
            model=self.model,
            platform=self.platform,
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
                model=self.model,
                platform=self.platform,
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
    # Example using HuggingFace platform
    llm = RagoonBot(model="meta-llama/Llama-3.2-3B-Instruct", platform="huggingface")
    print(llm.complete("Hello, how are you?").text)

    # Example using Together platform
    # llm = RagoonBot(model="some-together-model", platform="together")
    # print(llm.complete("Hello, how are you?").text)
