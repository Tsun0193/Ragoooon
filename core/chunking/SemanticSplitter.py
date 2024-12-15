from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

class SentenceSplitter(SemanticSplitterNodeParser):
    def __init__(self,
                 buffer_size: int = 512,
                 embed_model: str = None,
                 show_progress: bool = True,
                 include_metadata: bool = True):
        """
        Initializes the SemanticSplitter.
        
        :param buffer_size: int, default 512. Size of the buffer for text processing.
        :param embed_model: str, optional, default None. If None, OpenAIEmbedding is used. 
                             Otherwise, a HuggingFace embedding model is instantiated.
        :param show_progress: bool, default True. Whether to show progress during processing.
        :param include_metadata: bool, default True. Whether to include metadata in the output.
        """
        self.embed_model = OpenAIEmbedding() if embed_model is None else HuggingFaceEmbedding(model_name=embed_model)
        
        super().__init__(
            buffer_size=buffer_size,
            embed_model=embed_model,
            show_progress=show_progress,
            include_metadata=include_metadata
        )

    def process(self, text: str):
        """
        Splits the input text into chunks.

        :param text: str. The text to be split into chunks.
        :return: list. A list of text chunks.
        """
        document = Document(text=text)
        chunks = self.splitter.get_nodes_from_documents([document])
        return [chunk.text for chunk in chunks]