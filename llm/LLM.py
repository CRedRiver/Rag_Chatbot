from google import genai
from typing import List, Dict
from llm.config import SYS_INSTRUCTION

class LLM():
    def __init__(self, api_key:str, model_name:str="gemini-2.5-flash",
                 sys_instruction:str=SYS_INSTRUCTION):
        self.client = genai.Client(api_key = api_key)
        self.model_name = model_name
        self.instruction = sys_instruction
    
    def create_content(self, prompt, thinking_level="low"):
        interaction = self.client.interactions.create(
            model = self.model_name,
            system_instruction=self.instruction,
            input = prompt,
            generation_config = {
                "thinking_level": thinking_level
            }
        )
        return interaction.output_text

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    LLM = LLM(api_key = key)
    text = LLM.create_content("Explain AI in a few words")
    print(text)