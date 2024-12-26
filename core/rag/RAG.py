from core.llm.CustomLLM import RagoonBot
from dotenv import load_dotenv
from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.cortex import Complete
from llama_index.core.llms import LLM
from typing import Any, List, Dict


load_dotenv('../../.env')
llm = RagoonBot()


class Rag:
    def __init__(
        self, 
        llm: LLM = llm,
        transformers: Any = None, #TODO: add default transformers
        snowpark_session: Session = None, #TODO
        limit_to_retrieve: int = 4,
        snowflake_params: Dict[str, str] = {},
        search_columns: List[str] = ["NAME", "INFORMATION"],
        retrieve_column: str = "INFORMATION",
        
    ):
        """
        Initialize the RAG instance.

        TODO: param
        """
        if isinstance(llm, str):
            self.llm = RagoonBot(model=llm)
        else:
            self.llm = llm
        self.transformers = transformers
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve
        self.snowflake_params = snowflake_params
        self.search_columns = search_columns
        self.retrieve_column = retrieve_column


    def retrieve(self, query: str) -> List[str]:
        root = Root(self._snowpark_session)
        cortex_search_service = (
            root.databases[self.snowflake_params["SNOWFLAKE_DATABASE"]]
            .schemas[self.snowflake_params["SNOWFLAKE_SCHEMA"]]
            .cortex_search_services[self.snowflake_params["SNOWFLAKE_CORTEX_SEARCH_SERVICE"]]
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
    ):
        context = "\n\n".join(contexts)
        prompt = ( #TODO: prompting
        "You are an assistant for tourism and travel tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + query
    )

        response = self.llm.complete(prompt)
        return response

    def complete(
        self,
        text: str
    ):
        if self.transformers is not None:
            for transformer in self.transformers:
                text = transformer.transform(text)

        retrieved_contexts = self.retrieve(text)
        response = self.generate_response(retrieved_contexts, text)

        return response
        