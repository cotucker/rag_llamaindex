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

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# load documents
documents = SimpleDirectoryReader("data/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("What is closed-domain datasets")

print(response)
