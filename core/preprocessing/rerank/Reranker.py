from llama_index.core.postprocessor import LLMRerank
from llama_index.core.llms import LLM
from core.llm.CustomLLM import RagoonBot
from typing import List, Tuple, Union, Optional
from pydantic import PrivateAttr

from dotenv import load_dotenv

load_dotenv('../../.env')

llm = RagoonBot()

RERANK_PROMPT = f"""
Please rate the relevance of the following response to the given reference on a scale of 0 to 1. \n
Reference: REFERENCE_STRING\n
Response: INPUT_STRING\n
Return only the Relevance Score (0-1). Do not include any additional text.
"""

class Reranker(LLMRerank):
    """Reranker class for RagoonBot."""

    _original_string: str = PrivateAttr()

    def __init__(self, 
                 original_string: str,
                 llm: Union[LLM, str] = llm, 
                 top_n: int = 10, 
                 **kwargs):
        """
        Initialize the Reranker.

        :param original_string: The reference string to rerank against.
        :param llm: The RagoonBot instance to use for scoring.
        :param top_n: The number of top-ranked strings to return.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(llm=llm, top_n=top_n)
        self._original_string = original_string
        self.llm = llm
        self.top_n = top_n

    def get_relevant_score(self,
                           input_string: str,
                           reference_string: str) -> float:
        """
        Get the relevance score of the input string with respect to the reference string.

        :param input_string: The input string to score.
        :param reference_string: The reference string to score against.
        :return: The relevance score of the input string (0.0 to 1.0).
        """

        assert input_string is not None, "Please provide an input string to score."
        assert reference_string is not None, "Please provide a reference string to score against."

        prompt = RERANK_PROMPT.replace("REFERENCE_STRING", reference_string).replace("INPUT_STRING", input_string)
        
        try:
            response = self.llm.complete(prompt)
            score_str = response.text.strip().split('\n')[0]
            score = float(score_str)
            score = max(0.0, min(1.0, score))
        except (ValueError, IndexError, AttributeError) as e:
            print(f"Error parsing relevance score: {e}. Defaulting score to 0.0.")
            score = 0.0

        return score

    def transform(self, 
                  inputs: Union[str, List[str], List[List[str]]],
                  **kwargs) -> List[str]:
        """
        Rerank the input strings based on their relevance scores from RagoonBot.

        :param inputs: A string or a list of strings to be reranked.
        :return: A list of strings ordered from highest to lowest relevance.
        """
        assert inputs is not None, "Please provide input strings to rerank."

        # Normalize inputs to a flat list of strings
        if isinstance(inputs, str):
            _inputs = [inputs]
        elif isinstance(inputs, list):
            if not inputs:
                return []
            if isinstance(inputs[0], str):
                _inputs = inputs
            elif isinstance(inputs[0], list):
                _inputs = [item for sublist in inputs for item in sublist]
            else:
                raise ValueError("Invalid input format for reranking. Expected strings or list of strings.")
        else:
            raise ValueError("Invalid input type for reranking. Expected string or list of strings.")

        scores: List[Tuple[str, float]] = []

        for input_str in _inputs:
            score = self.get_relevant_score(input_str, self._original_string)
            scores.append((input_str, score))

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        top_n = min(self.top_n, len(sorted_scores))
        top_ranked = [s[0] for s in sorted_scores[:top_n]]

        return [top_ranked]

if __name__ == "__main__":
    ori = "What are the effects of schizophrenia on memory?"
    inputs = ["Schizophrenia is a severe mental disorder that affects how a person thinks, feels, and behaves.",
              "Schizophrenia can have a significant impact on memory and cognitive function.",
              "The effects of schizophrenia on memory can vary from person to person.",
              "Schizophrenia is associated with memory deficits and cognitive impairments.",
              "Memory loss is a common symptom of schizophrenia."]
    
    reranker = Reranker(original_string=ori)
    top_ranked = reranker.transform(inputs)
    print(top_ranked)