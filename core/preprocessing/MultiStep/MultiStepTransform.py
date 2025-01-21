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

        self.max_queries = 3

    def transform(self, text: Union[str, List[str], List[List[str]]],
                  max_queries: int = 3,
                  **kwargs) -> List[str]:
        """
        Decomposes the input query into multiple sub-queries (steps) for a step-by-step answer.
        Each sub-query should be separated by a newline.

        :param text: str. The text to decompose.
        :return: List of sub-queries.
        """
        # Prompt the model to break down the input query into sub-queries
        if isinstance(text, str):
            _text = [text]

        if isinstance(text[0], list) and len(text) == 1:
            _text = text[0]

        decomposition_prompts = [f"Please break down the question '{_t}' into smaller sub-queries that can be answered one by one. No more than {max_queries} sub-queries." for _t in _text]

        try:
            decompositions = [self.llm.complete(decomposition_prompt, temperature=0.2) for decomposition_prompt in decomposition_prompts]
            decompositions = [decomposition.text for decomposition in decompositions]
        except Exception as e:
            raise Exception(f"Error decomposing the input query: {str(e)}")

        try:
            sub_queries = [decomposition.split("\n")[1:] for decomposition in decompositions]
            sub_queries = [[sub_query for sub_query in sub_query_list if sub_query] for sub_query_list in sub_queries]
        except Exception as e:
            raise Exception(f"Error extracting sub-queries: {str(e)}")
        
        return sub_queries


if __name__ == "__main__":
    text = [["What are the effects of schizophrenia on memory?"]]
    transformer = MultiStepTransformer()

    print(transformer.transform(text))