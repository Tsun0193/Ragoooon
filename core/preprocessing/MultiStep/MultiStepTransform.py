import os
from llama_index.llms.huggingface import HuggingFaceLLM
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv('../../.env')

# Load Hugging Face Token
HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")

# Initialize the Hugging Face LLM with the appropriate model
llm = HuggingFaceLLM(model_name="meta-llama/Llama-3.2-3B-Instruct",
                     tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",)

class MultiStepTransformer:
    def __init__(self, llm: HuggingFaceLLM = None):
        """
        Initializes the MultiStepTransformer.

        :param llm: HuggingFaceLLM, default None. The LLM model to use.
        """
        self.llm = llm or HuggingFaceLLM(model_name="meta-llama/Llama-3.2-3B-Instruct",
                                        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct")

    def decompose_query(self, text: str) -> List[str]:
        """
        Decomposes the input query into multiple sub-queries (steps) for a step-by-step answer.
        Each sub-query should be separated by a newline.

        :param text: str. The text to decompose.
        :return: List of sub-queries.
        """
        # Prompt the model to break down the input query into sub-queries
        decomposition_prompt = f"Please break down the question '{text}' into smaller sub-queries that can be answered one by one."

        decomposition = self.llm.complete(decomposition_prompt, temperature=0.2)
        decomposition = decomposition.choices[0].text.strip()

        # Clean the decomposition output to get individual sub-queries
        sub_queries = decomposition.split('\n')
        sub_queries = [sub_query.strip() for sub_query in sub_queries if sub_query.strip()]
        return sub_queries

    def transform(self, text: str) -> str:
        """
        Transforms the input text into multi-step decomposed texts by interacting with the LLM model.
        
        :param text: str. The text to transform.
        :return: Combined responses from each sub-query.
        """
        if not text:
            return "Please provide a text to transform."

        # Step 1: Decompose the query into smaller sub-queries
        decomposed_queries = self.decompose_query(text)
        
        # Step 2: Generate answers for each sub-query
        results = []
        for query in decomposed_queries:
            if query:  # Avoid empty queries
                response = self.llm.generate(f"Answer the following query: {query}")
                results.append(f"Sub-query: {query} -> Answer: {response}")

        # Combine all sub-query results into one final response
        final_response = "\n".join(results)
        return final_response


# Example usage
text = "What are the effects of schizophrenia on memory?"
transformer = MultiStepTransformer(llm=llm)

print(transformer.transform(text))