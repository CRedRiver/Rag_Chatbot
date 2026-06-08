from embeddings.base import APIBaseEmbedding
from typing import Optional

import os
from dotenv import load_dotenv
from typing import List
load_dotenv()

class GeminiEmbedding(APIBaseEmbedding):
    def __init__(
        self,
        name: str = "text-embedding-2",
        api_key: Optional[str] = None,
    ):
        super().__init__(name=name, api_key=api_key)

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found! You must provide your own API key. "
                "Pass it as `api_key='...'` or set 'GEMINI_API_KEY' in your .env file."
            )

        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "To use Gemini embedding models, please run: `pip install google-genai`"
            )
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Verify the key actually works
        self._validate_api_key()
    
    def encode(self, contents: List[str], is_query: bool = False) -> List[List[float]]:
        if not contents:
            return []
        try:
            from google.genai import types
            current_task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
            result = self.client.models.embed_content(
                model=self.name,
                contents=contents,
                config=types.EmbedContentConfig(task_type=current_task)
            )
            return [embedding.values for embedding in result.embeddings]
        except Exception as e:
            raise ValueError(f"Gemini API call failed. Error: {e}") from e
    
    def _validate_api_key(self):
        """Pings the API with a 1-token string to ensure the key is valid and has quota."""
        try:
            from google.genai import types
            self.client.models.embed_content(
                model=self.name,
                contents=["test"],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            print("Gemini API Key validated successfully.")
        except Exception as e:
            raise PermissionError(
                f"Gemini API Key validation failed! Ensure your key is correct or has billing enabled. Error: {e}"
            ) from e