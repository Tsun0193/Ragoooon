import os
import warnings
from core.llm.CustomLLM import RagoonBot
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.llms import LLM
from snowflake.snowpark import Session
from snowflake.cortex import Complete
from dotenv import load_dotenv
from typing import Any

warnings.filterwarnings("ignore")
load_dotenv("../../.env")

llm = RagoonBot()

class HyDETransformer(HyDEQueryTransform):
    def __init__(self, 
                 llm: LLM = llm,
                 hyde_prompt: str = None,
                 include_original: bool = True):
        """
        Initializes the Hypothetical Document Embeddings 

        :param llm: str, default None. The LLM model to use.
        :param hyde_prompt: str, default None. The prompt to use for the HyDE model.
        :param include_original: bool, default True. Whether to include the original text in the output.
        """
        if isinstance(llm, str):
            self.llm = RagoonBot(model=llm)
        else:
            self.llm = llm
            
        super().__init__(
            llm=self.llm,
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
    
if __name__ == "__main__":
    transformer = HyDETransformer()
    response = transformer.transform(text="Hello, how are you?")
    print(response)