import os
import shutil
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cerebras import Cerebras
from dotenv import load_dotenv
from config import settings

load_dotenv()

CEREBRAS_API_KEY = settings.llm.api_key
if not CEREBRAS_API_KEY:
    raise ValueError("CEREBRAS_API_KEY environment variable is not set.")

Settings.llm = Cerebras(model=settings.llm.model_name, api_key=CEREBRAS_API_KEY)
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L12-v2")

def get_document_from_pdf(path_to_pdf: str) -> Document:
    import pymupdf
    doc = pymupdf.open(path_to_pdf)
    text = '\n'.join([page.get_text() for page in doc])
    return Document(text=text, metadata={"file_path": path_to_pdf, "file_name": os.path.basename(path_to_pdf)})

def get_document_from_txt(path_to_txt: str) -> Document:
    with open(path_to_txt, "r", encoding="utf-8") as f:
        text = f.read()
    return Document(text=text, metadata={"file_path": path_to_txt, "file_name": os.path.basename(path_to_txt)})

def get_documents(path: str) -> list[Document]:
    documents = []
    if not os.path.exists(path):
        os.makedirs(path)
        return []

    print(f"📂 Scanning folder: {path}")
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        try:
            if filename.lower().endswith(".pdf"):
                documents.append(get_document_from_pdf(full_path))
                print(f"   - Added PDF: {filename}")
            elif filename.lower().endswith(".txt"):
                documents.append(get_document_from_txt(full_path))
                print(f"   - Added TXT: {filename}")
        except Exception as e:
            print(f"   ❌ Error reading file {filename}: {e}")
    return documents

def initialize_index():
    db_path = settings.vector_store.path
    collection_name = settings.vector_store.collection_name
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(name="quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        print(f"💾 Found existing database ({chroma_collection.count()} chunks). Loading...")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
    else:
        print("🆕 Database is empty or not found. Creating index...")
        documents = get_documents(settings.domain.domain_path)

        if not documents:
            print("⚠️ No documents to index! Please place files in the data/ folder")
            return VectorStoreIndex.from_documents([], storage_context=storage_context, embed_model=embed_model)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model
        )
        print("✅ Indexing complete and saved!")

    return index

_index_instance = initialize_index()

def get_response(query_text: str):
    query_engine = _index_instance.as_query_engine(similarity_top_k=20)
    response = query_engine.query(query_text)
    return response
