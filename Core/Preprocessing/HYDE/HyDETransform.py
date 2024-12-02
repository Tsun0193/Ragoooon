from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from dotenv import load_dotenv

load_dotenv()

class HyDETransformer(HyDEQueryTransform):
    def __init__(self, 
                 llm: str = None,
                 hyde_prompt: str = None,
                 include_original: bool = True):
        """
        Initializes the Hypothetical Document Embeddings 

        :param llm: str, default None. The LLM model to use.
        :param hyde_prompt: str, default None. The prompt to use for the HyDE model.
        :param include_original: bool, default True. Whether to include the original text in the output.
        """
        super().__init__(
            llm=llm,
            hyde_prompt=hyde_prompt,
            include_original=include_original
        )
        

    def transform(
        self,
        text: str = None
    ):
        """
        Transforms the input text into hypothetical document embeddings.

        :param text: str. The text to transform.
        :return: str. The transformed text.
        """
        if text is None:
            return "Please provide a text to transform."
        
        response = self.run(text)
        return response.custom_embedding_strs
    

text = "What are the effects of schizophrenia on memory?"
transformer = HyDETransformer()

print(transformer.transform(text))