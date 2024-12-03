from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform
from dotenv import load_dotenv

load_dotenv()

class MultiStepTransformer(StepDecomposeQueryTransform):
    def __init__(self,
                 llm: str = None):
        """
        Initializes the MultiStepTransformer.

        :param llm: str, default None. The LLM model to use. By default, the model is OpenAI's GPT-3.
        """
        super().__init__(
            llm = llm
        )

    def transform(self,
                  text: str = None):
        """
        Transforms the input text into multi-step decomposed texts.
        
        :param text: str. The text to transform.
        """
        if text is None:
            return "Please provide a text to transform."
        
        response = self.run(text)
        return response
    
text = "What are the effects of schizophrenia on memory?"
transformer = MultiStepTransformer()

print(transformer.transform(text))