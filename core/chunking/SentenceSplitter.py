from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

class SentenceSplitter:
    def __init__(self,
                 chunk_size: int = 64,
                 chunk_overlap: int = 32):
        """
        Initializes the SentenceSplitter.

        :param chunk_size: int, default 64. The size of each chunk.
        :param chunk_overlap: int, default 32. The size of overlap between chunks.
        """
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
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
