from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cerebras import Cerebras
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()
def get_document_from_pdf(path_to_pdf: str) -> Document:
    import pymupdf
    doc = pymupdf.open(path_to_pdf)
    text = '\n'.join([page.get_text() for page in doc])
    return Document(text=text, metadata={"file_path": path_to_pdf})

def get_document_from_txt(path_to_txt: str) -> Document:
    with open(path_to_txt, "r", encoding="utf-8") as f:
        text = f.read()
    return Document(text=text, metadata={"file_path": path_to_txt})

def get_domuments(path: str) -> list[Document]:
    documents = []
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if filename.lower().endswith(".pdf"):
            documents.append(get_document_from_pdf(full_path))
        elif filename.lower().endswith(".txt"):
            documents.append(get_document_from_txt(full_path))
    return documents

def print_response_sources(response):
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

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise ValueError("CEREBRAS_API_KEY environment variable is not set.")

Settings.llm = Cerebras(model="gpt-oss-120b", api_key=CEREBRAS_API_KEY)

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L12-v2")

# documents = SimpleDirectoryReader("data/").load_data()
documents = get_domuments("data/")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What is ownership and borrowing in Rust?")

print(f"Response: {response}\n")

# print_response_sources(response)
