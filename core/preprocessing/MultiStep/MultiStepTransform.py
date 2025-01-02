import os
from core.llm.CustomLLM import RagoonBot
from llama_index.core.llms import LLM
from dotenv import load_dotenv
from typing import Optional, List, Union, Callable

load_dotenv('../../.env')

llm = RagoonBot()

class MultiStepTransformer:
    def __init__(self, llm: Union[LLM, str] = llm):
        """
        Initializes the MultiStepTransformer.

        :param llm: LLM, default None. The LLM model to use.
        """
        if isinstance(llm, str):
            try:
                self.llm = RagoonBot(model=llm)
            except Exception as e:
                raise Exception(f"Error initializing LLM model: {str(e)}")
        else:
            self.llm = llm

    def transform(self, text: str, max_queries: int = 5,
                  **kwargs) -> List[str]:
        """
        Decomposes the input query into multiple sub-queries (steps) for a step-by-step answer.
        Each sub-query should be separated by a newline.

        :param text: str. The text to decompose.
        :return: List of sub-queries.
        """
        # Prompt the model to break down the input query into sub-queries
        decomposition_prompt = f"Please break down the question '{text}' into smaller sub-queries that can be answered one by one. No more than {max_queries} sub-queries."

        try:
            decomposition = self.llm.complete(decomposition_prompt, temperature=0.2)
            decomposition = decomposition.text
        except Exception as e:
            raise Exception(f"Error decomposing the input query: {str(e)}")

        # Clean the decomposition output to get individual sub-queries
        try:
            sub_queries = decomposition.split('\n')
            sub_queries = [sub_query.strip() for sub_query in sub_queries if sub_query.strip()]
            sub_queries = sub_queries[1:]  # Remove the first sub-query which is the prompt
        except Exception as e:
            raise Exception(f"Error processing the decomposition output: {str(e)}")
        
        return sub_queries


if __name__ == "__main__":
    text = "What are the effects of schizophrenia on memory?"
    transformer = MultiStepTransformer()

    print(transformer.transform(text))