from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser, LanguageConfig
from llama_index.core import Document

config = LanguageConfig(language="english", spacy_model="en_core_web_md")

class SemanticDoubleMergingSplitter(SemanticDoubleMergingSplitterNodeParser):
    def __init__(self, 
                 language_config: LanguageConfig = config,
                 initial_threshold: float = 0.5,
                 appending_threshold: float = 0.7,
                 merging_threshold: float = 0.8,
                 max_chunk_size: int = 512):
        """
        Initializes the SemanticDoubleMergingSplitter.

        :param language_config: LanguageConfig, default config. The language configuration for processing.
        :param initial_threshold: float, default 0.5. The threshold for initial chunking.
        :param appending_threshold: float, default 0.7. The threshold for appending chunks.
        :param merging_threshold: float, default 0.8. The threshold for merging chunks.
        :param max_chunk_size: int, default 512. The maximum size of a chunk.
        """
        super().__init__(
            language_config=language_config,
            initial_threshold=initial_threshold,
            appending_threshold=appending_threshold,
            merging_threshold=merging_threshold,
            max_chunk_size=max_chunk_size
        )

    def process(self, text: str):
        """
        Splits the input text into chunks based on the defined thresholds.

        :param text: str. The text to be split into chunks.
        :return: list. A list of text chunks.
        """
        document = Document(text=text)
        chunks = self.get_nodes_from_documents([document])
        return [chunk.text for chunk in chunks]