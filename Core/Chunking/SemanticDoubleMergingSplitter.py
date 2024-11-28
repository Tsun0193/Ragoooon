import pandas as pd
import nltk
from snowflake.snowpark.types import StringType, StructField, StructType
from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser, LanguageConfig
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Document

config = LanguageConfig(language="english", spacy_model="en_core_web_md")

class SemanticDoubleMergingSplitter:
    def __init__(self, 
                 language_config: LanguageConfig = config,
                 initial_threshold: float = 0.5,
                 appending_threshold: float = 0.7,
                 merging_threshold: float = 0.8,
                 max_chunk_size: int = 512):
        self.splitter = SemanticDoubleMergingSplitterNodeParser(
            language_config=language_config,
            initial_threshold=initial_threshold,
            appending_threshold=appending_threshold,
            merging_threshold=merging_threshold,
            max_chunk_size=max_chunk_size
        )

    def process(self, _text: str):
        document = Document(text=_text)
        chunks = self.splitter.get_nodes_from_documents([document])
        return [chunk.text for chunk in chunks]
