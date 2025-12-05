import os
import json
import shutil
import chromadb
import pymupdf
import hashlib
from typing import List, Dict, Tuple
from llama_index.core import VectorStoreIndex, Settings, Document, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cerebras import Cerebras
from dotenv import load_dotenv
from src.config import settings
from src.doc_parser import get_images_description

load_dotenv()

STATE_FILE = "doc_state.json"
CEREBRAS_API_KEY = settings.llm.api_key

if not CEREBRAS_API_KEY:
    raise ValueError("CEREBRAS_API_KEY environment variable is not set.")

Settings.llm = Cerebras(model=settings.llm.model_name, api_key=CEREBRAS_API_KEY)
embed_model = HuggingFaceEmbedding(model_name=settings.embedding.model_name, trust_remote_code=True)

def get_document_from_pdf(path_to_pdf: str) -> Document:
    doc = pymupdf.open(path_to_pdf)
    text = '\n'.join([page.get_text() + '\n' + get_images_description(page) for page in doc])
    return Document(text=text, metadata={"file_path": path_to_pdf, "file_name": os.path.basename(path_to_pdf)})

def get_document_from_txt(path_to_txt: str) -> Document:
    with open(path_to_txt, "r", encoding="utf-8") as f:
        text = f.read()
    return Document(text=text, metadata={"file_path": path_to_txt, "file_name": os.path.basename(path_to_txt)})

def parse_specific_files(file_paths: List[str]) -> List[Document]:
    documents = []
    for full_path in file_paths:
        if not os.path.exists(full_path):
            continue

        filename = os.path.basename(full_path)
        try:
            if filename.lower().endswith(".pdf"):
                documents.append(get_document_from_pdf(full_path))
                print(f"   - Parsed PDF: {filename}")
            elif filename.lower().endswith(".txt"):
                documents.append(get_document_from_txt(full_path))
                print(f"   - Parsed TXT: {filename}")
        except Exception as e:
            print(f"   ❌ Error reading file {filename}: {e}")
    return documents

def get_file_hash(filepath: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_current_files_state(folder_path: str) -> Dict[str, str]:
    state = {}
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return state

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.pdf', '.txt', '.md', '.docx')):
            full_path = os.path.join(folder_path, filename)
            state[filename] = get_file_hash(full_path)
    return state

def check_for_updates():
    current_state = get_current_files_state(settings.domain.domain_path)
    saved_state = {}

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            saved_state = json.load(f)

    new_files = []
    modified_files = []
    deleted_files = []

    for filename, file_hash in current_state.items():

        if filename not in saved_state:
            new_files.append(filename)
        elif saved_state[filename] != file_hash:
            modified_files.append(filename)

    for filename in saved_state:

        if filename not in current_state:
            deleted_files.append(filename)

    return {
        "new": new_files,
        "modified": modified_files,
        "deleted": deleted_files,
        "current_state": current_state
    }

def get_vector_store():
    db = chromadb.PersistentClient(path=settings.vector_store.path)
    chroma_collection = db.get_or_create_collection(name=settings.vector_store.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store, chroma_collection

def apply_updates(updates: Dict):
    vector_store, chroma_collection = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    files_to_remove = updates['deleted'] + updates['modified']
    files_to_add = updates['new'] + updates['modified']

    if files_to_remove:
        print(f"🗑️  Removing chunks for: {', '.join(files_to_remove)}")
        chroma_collection.delete(where={"file_name": {"$in": files_to_remove}})

    if files_to_add:
        print(f"📥 Processing new/modified files: {', '.join(files_to_add)}")
        full_paths = [os.path.join(settings.domain.domain_path, f) for f in files_to_add]
        new_documents = parse_specific_files(full_paths)

        if new_documents:
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=70,
                embed_model=embed_model
            )
            nodes = splitter.get_nodes_from_documents(new_documents)
            index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
            index.insert_nodes(nodes)
            print(f"✅ Added {len(nodes)} new chunks.")

    with open(STATE_FILE, 'w') as f:
        json.dump(updates['current_state'], f)

    print("💾 State updated.")

def initialize_index():
    vector_store, chroma_collection = get_vector_store()

    if chroma_collection.count() == 0 and not os.path.exists(STATE_FILE):
        print("🆕 First run detected. Scanning documents...")
        updates = check_for_updates()

        if updates['new']:
            apply_updates(updates)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index

_index_instance = None

def get_query_engine():
    global _index_instance
    if _index_instance is None:
        _index_instance = initialize_index()
    return _index_instance.as_query_engine(similarity_top_k=settings.vector_store.top_k)

def get_response(query_text: str):
    engine = get_query_engine()
    response = engine.query(query_text)
    return response
