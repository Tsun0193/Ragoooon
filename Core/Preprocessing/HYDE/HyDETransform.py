import os
import warnings
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata, LLM
from llama_index.core.llms.callbacks import llm_completion_callback
from huggingface_hub import InferenceClient
from snowflake.snowpark import Session
from snowflake.cortex import Complete
from dotenv import load_dotenv
from typing import Any

warnings.filterwarnings("ignore")
load_dotenv()

connection_params = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
}
snowflake_session = Session.builder.configs(connection_params).create()

client = InferenceClient(api_key=os.environ["HF_TOKEN"])
def complete(user_text, model = "meta-llama/Llama-3.2-3B-Instruct"):
    # completion = Complete(
    #     model="snowflake-arctic",
    #     prompt=user_text,
    #     session=snowflake_session,
    # )
    # return completion

    messages = [
        {
            "role": "user",
            "content": user_text
        }
    ]

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

class HyDETransformer(HyDEQueryTransform):
    def __init__(self, 
                 llm: LLM,
                 hyde_prompt: str = None,
                 include_original: bool = True):
        """
        Initializes the Hypothetical Document Embeddings 

        :param llm: str, default None. The LLM model to use.
        :param hyde_prompt: str, default None. The prompt to use for the HyDE model.
        :param include_original: bool, default True. Whether to include the original text in the output.
        """
        super().__init__(
            llm=llm,
            hyde_prompt=hyde_prompt,
            include_original=include_original
        )
        

    def transform(
        self,
        text: str = None
    ):
        """
        Transforms the input text into hypothetical document embeddings.

        :param text: str. The text to transform.
        :return: str. The transformed text.
        """
        if text is None:
            return "Please provide a text to transform."
        
        response = self.run(text)
        return response.custom_embedding_strs
    
llm = RagoonBot()
text = "What are the effects of schizophrenia on memory?"
transformer = HyDETransformer(
    llm=llm
)

for query in transformer.transform(text):
    print(query)
    print("\n")
    print("-" * 100)
    print("\n")