from core.llm.CustomLLM import RagoonBot
import os
from dotenv import load_dotenv
from snowflake.snowpark.session import Session
from snowflake.core import Root
from snowflake.cortex import Complete
from llama_index.core.llms import LLM
from typing import Any, List, Dict


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

class Rag:
    def __init__(
        self, 
        llm: LLM = llm,
        transformers: Any = None, #TODO: add default transformers
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
        self.transformers = transformers
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve
        self.snowflake_params = snowflake_params
        self.search_columns = search_columns
        self.retrieve_column = retrieve_column


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
        prompt: str,
        **kwargs
    ):
        if self.transformers is not None:
            for transformer in self.transformers:
                prompt = transformer.transform(prompt)

        retrieved_contexts = self.retrieve(prompt)
        response = self.generate_response(retrieved_contexts, prompt)

        return response
        

if __name__ == "__main__":
    rag = Rag(
        llm=llm,
        transformers=None,
        snowpark_session=snowpark_session,
        snowflake_params=connection_params
    )
    
    response = rag.complete("Where should I eat in Hanoi?")
    print(response)

    if snowpark_session:
            # Close the Snowflake session
        snowpark_session.close()
