import os
import time

from dotenv import load_dotenv
from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.cortex import Complete
from llama_index.core.llms import LLM
from typing import Any, List, Dict, Callable, Union, Optional, Tuple
from llama_index.core.llms import CompletionResponse, CompletionResponseGen
from core.llm.CustomLLM import RagoonBot
from core.preprocessing.HYDE.HyDETransform import HyDETransformer
from core.preprocessing.MultiStep.MultiStepTransform import MultiStepTransformer
from core.preprocessing.rerank.Reranker import Reranker
from geo.utils import *

load_dotenv('../../.env')
llm = RagoonBot()

try:
    connection_params = {
        "account":  os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
        "role": os.environ["SNOWFLAKE_ROLE"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
        "schema": os.environ["SNOWFLAKE_SCHEMA"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "service": os.environ["SNOWFLAKE_CORTEX_SEARCH_SERVICE"],
    }
except KeyError as e:
    print("Please set the environment variable: " + str(e))

try:
    snowpark_session = Session.builder.configs(connection_params).create()
except Exception as e:
    print("Error creating Snowpark session: " + str(e))

transforms = {
    'HyDE': HyDETransformer(),
    'MultiStep': MultiStepTransformer(),
    'Rerank': Reranker("")
}

class Rag:
    def __init__(
        self, 
        llm: LLM = llm,
        transformers: Union[str, List[str]] = ["MultiStep", "HyDE"],
        snowpark_session: Session = snowpark_session,
        limit_to_retrieve: int = 4,
        snowflake_params: Dict[str, str] = connection_params,
        search_columns: List[str] = ["NAME", "INFORMATION"],
        retrieve_column: str = "INFORMATION"
    ):
        """
        Initialize the RAG instance.

        TODO: param
        """
        if isinstance(llm, str):
            self.llm = RagoonBot(model=llm)
        else:
            self.llm = llm
        
        if isinstance(transformers, str):
            self.transformers = [transformers]
        else:
            self.transformers = transformers

        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve
        self.snowflake_params = snowflake_params
        self.search_columns = search_columns
        self.retrieve_column = retrieve_column

    def controller(self, text: str, **kwargs: Any) -> bool:
        # If the text is about basic information, return True
        prompt = """
            Return True if the user query is about basic information or general knowledge.
            Otherwise, return False.
            Do not include any other information, just True or False.

            User Query: {}
        """
        response = self.llm.complete(prompt.format(text))
        response = response.text
        response = response.strip()
        assert response in ["True", "False"], f"Invalid response from the controller: {response.text}"
        return response == "True"

    def retrieve(self, query: str) -> List[str]:
        root = Root(self._snowpark_session)
        cortex_search_service = (
            root.databases[self.snowflake_params.get("database")]
            .schemas[self.snowflake_params.get("schema")]
            .cortex_search_services[self.snowflake_params.get("service")]
        )
        resp = cortex_search_service.search(
            query=query,
            columns=self.search_columns,
            limit=self._limit_to_retrieve,
        )

        if resp.results:
            return [curr[self.retrieve_column] for curr in resp.results]
        else:
            return []
    

    def generate_response(
        self,
        contexts: List[str] = [],
        query: str = None,
        history: Optional[List[dict]] = None,
        **kwargs: Any
    ):
        assert query is not None, "Query cannot be None."
        if not contexts:
            context = "None"

        context = "\n\n".join(contexts)
        # Combine history with the user prompt
        # start with assistant introducing itself
        history_text = "Assistant: Hello, I am Ragoon, an assistant for tourism and travel tasks."
        if history:
            history_text += "\n".join([f"{entry['role']}: {entry['content']}" for entry in history])

        prompt = f"""
            These are the messages between a user and an assistant:
            {history_text}
            Now, use the following pieces of retrieved context to complete the conversation by answering the user's question. 
            If you don't know the answer, say that you don't know. 
            Keep the answer concise.
        """
        prompt += f"\n\nContext: {context} \n\nQuery: {query}"

        response = self.llm.complete(prompt)
        # print(prompt)
        return response

    def complete(
        self,
        prompts: Union[str, List[str]] = None,
        history: Optional[List[dict]] = None,
        **kwargs
    ):
        """
        Completes the prompt using the RAG model.

        :param prompts: str. The prompts to complete.
        :return: str. The completed prompts.
        """
        assert prompts is not None, "Prompt cannot be None."
        
        if isinstance(prompts, str):
            _prompt = [[prompts]]
            original_prompt = prompts

        if isinstance(prompts, list):
            _prompt = [prompts]
            original_prompt = prompts[0]

        # Reduce transforms for basic queries
        if self.controller(original_prompt):
            # No need for transformers
            _prompt = [[original_prompt]]
        else:
            transforms["Rerank"]._original_string = original_prompt
            if self.transformers is not None:
                for _transformer in self.transformers:
                    prime = transforms.get(_transformer)
                    _prompt = prime.transform(_prompt)

        retrieved_contexts = []
        for _p in _prompt:
            retrieved_contexts.extend(self.retrieve(_p[0]))
        
        try:
            response = self.generate_response(
                contexts=retrieved_contexts,
                query=original_prompt,
                history=history
            )
        except Exception as e:
            return f"Error: {e}"
        
        return response.text

    def stream_complete(
        self,
        prompts: str,
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
            full_response = self.complete(prompts=prompts, history=history)
        except Exception as e:
            yield CompletionResponse(text="", delta=f"Error: {e}")
            return

        accumulated_text = ""
        for r in full_response:
            accumulated_text += r
            yield CompletionResponse(text=accumulated_text, delta=r)

if __name__ == "__main__":
    rag = Rag(
        llm=llm
    )
    
    response = rag.complete("Where should I eat in Hanoi?")
    print(response)

    if snowpark_session:
        # Close the Snowflake session
        snowpark_session.close()
