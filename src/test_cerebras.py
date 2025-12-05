import os
from dotenv import load_dotenv
from llama_index.llms.cerebras import Cerebras

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

llm = Cerebras(model="gpt-oss-120b", api_key=CEREBRAS_API_KEY)
response = llm.complete("What is Generative AI?")
print(response)
