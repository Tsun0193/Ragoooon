from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document

class SentenceWindowSplitter:
    def __init__(self,
                 window_size: int = 3,
                 window_metadata_key: str = "window",
                 original_text_metadata_key: str = "original_sentence"):
        """
        Initializes the SentenceWindowSplitter.

        :param window_size: int, default 3. The size of the window for splitting text.
        :param window_metadata_key: str, default "window". Key to store the window in the metadata.
        :param original_text_metadata_key: str, default "original_sentence". Key to store the original sentence in the metadata.
        """
        self.splitter = SentenceWindowNodeParser(
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key
        )

    def process(self, text: str):
        """
        Splits the input text into chunks using a sliding window approach.

        :param text: str. The text to be split into chunks.
        :return: list. A list of text chunks.
        """
        document = Document(text=text)
        chunks = self.splitter.get_nodes_from_documents([document])
        return [chunk.text for chunk in chunks]
