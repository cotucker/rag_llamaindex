from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cerebras import Cerebras
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise ValueError("CEREBRAS_API_KEY environment variable is not set.")

Settings.llm = Cerebras(model="gpt-oss-120b", api_key=CEREBRAS_API_KEY)

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L12-v2")

documents = SimpleDirectoryReader("data/").load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("TriviaQA of Joshi")

print(f"Response: {response}\n")

print("--- Source Documents ---")
for node_with_score in response.source_nodes:
    node = node_with_score.node
    text_content = node.get_text()
    file_path = node.metadata.get("file_path")

    start_line = "Unknown"
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
                start_index = full_text.find(text_content)
                if start_index != -1:
                    start_line = full_text.count('\n', 0, start_index) + 1
        except Exception:
            pass

    print(f"File: {file_path}")
    print(f"Line: {start_line}")
    print(f"Content snippet:\n{text_content}\n")
    print("-" * 30)
