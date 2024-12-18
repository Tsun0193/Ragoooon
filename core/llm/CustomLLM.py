import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.cortex import Complete
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

load_dotenv('../../.env')

connection_params = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
}

snowflake_session = Session.builder.configs(connection_params).create()

def complete(user_text: str,
             model: str = "mistral-large2",
             history: Optional[List[dict]] = None) -> str:
    """
    Perform a completion using Snowflake's Complete API.

    :param user_text: The input prompt for the model.
    :param model: The model to use for completion.
    :param history: Optional history of previous interactions.
    :return: The generated completion text.
    """
    completion = Complete(
        model=model,
        prompt=user_text,
        session=snowflake_session
    )
    
    return completion

class RagoonBot(CustomLLM):
    """
    RagoonBot is a custom LLM model that uses Snowflake's Complete API to generate text completions.

    :param model: str, default "mistral-large2". The model name to use.
    :param context_window: int, default 3900. The context window size.
    """
    model: str = "mistral-large2"

    def __init__(
        self, 
        model: str = "mistral-large2",
        **kwargs: Any
    ):
        """
        Initialize the RagoonBot instance.

        :param model: The model name to use for completions.
        :param context_window: The context window size.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model = model
        print(f"RagoonBot initialized with model: {self.model}")

    @property
    def metadata(self) -> LLMMetadata:
        """
        Provide metadata about the LLM.

        :return: An instance of LLMMetadata containing model information.
        """
        return LLMMetadata(
            num_output=1,
            model_name=self.model
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Generate a completion for the given prompt.

        :param prompt: The input text prompt.
        :param history: Optional history of previous interactions.
        :return: A CompletionResponse containing the generated text.
        """
        try:
            response_text = complete(
                user_text=prompt,
                model=self.model,
                history=history
            )
        except Exception as e:
            response_text = f"Error: {e}"

        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Generate a streamed completion for the given prompt.

        :param prompt: The input text prompt.
        :param history: Optional history of previous interactions.
        :yield: Partial CompletionResponses as text is generated.
        """
        try:
            full_response = complete(
                user_text=prompt,
                model=self.model,
                history=history
            )
        except Exception as e:
            yield CompletionResponse(text="", delta=f"Error: {e}")
            return

        accumulated_text = ""
        for char in full_response:
            accumulated_text += char
            yield CompletionResponse(text=accumulated_text, delta=char)

if __name__ == "__main__":
    llm = RagoonBot(model="mistral-large2")
    response = llm.complete("Hello, how are you?")
    print(response)
