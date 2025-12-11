import os
import shutil
import gc
import re
import json
import time
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterCondition, FilterOperator
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cerebras import Cerebras
from llama_index.llms.groq import Groq
from src.config import settings
from src.doc_parser import (
    get_document_from_pdf,
    get_document_from_txt,
    get_document_from_md,
    get_document_from_docx,
    get_document_from_csv,
    get_document_from_xlsx,
    get_document_from_image,
    get_images_description,
    set_access_control_config
)

LLM_API_KEY = settings.llm.api_key

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY environment variable is not set.")
match settings.llm.provider:
    case "cerebras":
        Settings.llm = Cerebras(model=settings.llm.model_name, api_key=LLM_API_KEY)
    case "groq":
        Settings.llm = Groq(model=settings.llm.model_name, api_key=LLM_API_KEY)
    case _:
        raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")

embed_model = HuggingFaceEmbedding(
    model_name=settings.embedding.model_name,
    trust_remote_code=True,
    model_kwargs={"attn_implementation": "sdpa"}
)
embed_chunking_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")
STATE_FILE = os.path.join(settings.vector_store.path, "kb_state.json")
ACCESS_CONTROL_FILE = "access_config.json"
ACCESS_CONTROL_STATUS = "private"

# def update_access_config(access_config: dict):



def sync_access_levels():
    print("🔄 Syncing access levels across the database...")
    db_path = settings.vector_store.path
    collection_name = settings.vector_store.collection_name
    config_path = ACCESS_CONTROL_FILE

    if not os.path.exists(config_path):
        print("⚠️ Access config not found. Skipping sync.")
        return

    try:
        with open(config_path, 'r') as f:
            access_config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading access config: {e}")
        return

    try:
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_collection(name=collection_name)
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return

    existing_data = collection.get(include=["metadatas"])
    ids_to_update = []
    metadatas_to_update = []
    update_count = 0
    total_chunks = len(existing_data["ids"])
    print(f"📊 Scanning {total_chunks} chunks...")

    for i, doc_id in enumerate(existing_data["ids"]):
        current_metadata = existing_data["metadatas"][i]  # ty:ignore[non-subscriptable]
        file_name = current_metadata.get("file_name")

        if not file_name:
            continue

        target_access = access_config.get(file_name, "public")
        current_access = current_metadata.get("access_level")

        if current_access != target_access:
            current_metadata["access_level"] = target_access  # ty:ignore[invalid-assignment]
            ids_to_update.append(doc_id)
            metadatas_to_update.append(current_metadata)
            update_count += 1

    if ids_to_update:
        print(f"📝 Updating {len(ids_to_update)} chunks...")
        batch_size = 5000

        for i in range(0, len(ids_to_update), batch_size):
            end = i + batch_size
            collection.update(
                ids=ids_to_update[i:end],
                metadatas=metadatas_to_update[i:end]
            )

        print(f"✅ Successfully updated access levels for {update_count} chunks.")
    else:
        print("✅ No access level changes required.")

def get_access_control_config():
    with open(ACCESS_CONTROL_FILE, 'r') as f:
        config = json.load(f)
    return config

def get_documents_access_control():
    path = settings.domain.domain_path
    documents = []
    if not os.path.exists(path):
        os.makedirs(path)
        return []

    print(f"📂 Scanning folder: {path}")
    filenames = sorted(
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )
    return filenames, get_access_control_config()

def save_access_control_config(config: dict):
    config_path = ACCESS_CONTROL_FILE
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

set_access_control_config(get_access_control_config())

def get_node_parser():
    def _simple_sentence_splitter(text: str):
        sentences = re.split(r'(?<=[\.\!\?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    return SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=70,
        sentence_splitter=_simple_sentence_splitter,
        embed_model=embed_chunking_model
    )

def get_documents(path: str):
    documents = []
    if not os.path.exists(path):
        os.makedirs(path)
        return []

    print(f"📂 Scanning folder: {path}")
    filenames = sorted(os.listdir(path))

    for filename in filenames:
        full_path = os.path.join(path, filename)
        try:
            if filename.lower().endswith(".pdf"):
                documents.append(get_document_from_pdf(full_path))
                print(f"   - Added PDF: {filename}")
            elif filename.lower().endswith(".txt"):
                documents.append(get_document_from_txt(full_path))
                print(f"   - Added TXT: {filename}")
            elif filename.lower().endswith(".md"):
                documents.append(get_document_from_md(full_path))
                print(f"   - Added MD: {filename}")
            elif filename.lower().endswith(".docx"):
                documents.append(get_document_from_docx(full_path))
                print(f"   - Added DOCX: {filename}")
            elif filename.lower().endswith(".csv"):
                documents.append(get_document_from_csv(full_path))
                print(f"   - Added CSV: {filename}")
            elif filename.lower().endswith(".xlsx"):
                documents.append(get_document_from_xlsx(full_path))
                print(f"   - Added XLSX: {filename}")
            elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                documents.append(get_document_from_image(full_path))
                print(f"   - Added IMAGE: {filename}")
        except Exception as e:
            print(f"   ❌ Error reading file {filename}: {e}")

    splitter = get_node_parser()
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def get_current_state(path: str):
    state = {}
    if not os.path.exists(path):
        return state
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path) and (filename.lower().endswith('.pdf')
            or filename.lower().endswith('.txt') or filename.lower().endswith('.csv') or filename.lower().endswith('.xlsx')
            or filename.lower().endswith('.md')) or filename.lower().endswith('.docx') or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
             state[filename] = os.path.getmtime(full_path)
    return state

def check_for_updates():
    domain_path = settings.domain.domain_path

    if not os.path.exists(STATE_FILE):
        current_state = get_current_state(domain_path)
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

        with open(STATE_FILE, 'w') as f:
            json.dump(current_state, f)

        return None

    with open(STATE_FILE, 'r') as f:
        try:
            saved_state = json.load(f)
        except json.JSONDecodeError:
            saved_state = {}

    current_state = get_current_state(domain_path)
    changes = {'added': [], 'modified': [], 'deleted': []}

    for filename, mtime in current_state.items():
        if filename not in saved_state:
            changes['added'].append(filename)
        elif abs(mtime - saved_state[filename]) > 1: # 1 second tolerance
            changes['modified'].append(filename)

    for filename in saved_state:
        if filename not in current_state:
            changes['deleted'].append(filename)

    if not any(changes.values()):
        return None

    return changes

def update_knowledge_base(changes):
    domain_path = settings.domain.domain_path
    db_path = settings.vector_store.path
    collection_name = settings.vector_store.collection_name
    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_collection(name=collection_name)
    files_to_delete = changes['deleted'] + changes['modified']

    for filename in files_to_delete:
        print(f"🗑️ Removing old chunks for: {filename}")
        collection.delete(where={"file_name": filename})

    files_to_add = changes['added'] + changes['modified']

    if files_to_add:
        print(f"🔄 Processing {len(files_to_add)} new/updated files...")
        new_documents = []

        for filename in files_to_add:
            full_path = os.path.join(domain_path, filename)
            try:
                if filename.lower().endswith(".pdf"):
                    new_documents.append(get_document_from_pdf(full_path))
                elif filename.lower().endswith(".txt"):
                    new_documents.append(get_document_from_txt(full_path))
                elif filename.lower().endswith(".md"):
                    new_documents.append(get_document_from_md(full_path))
                elif filename.lower().endswith(".docx"):
                    new_documents.append(get_document_from_docx(full_path))
                elif filename.lower().endswith(".csv"):
                    new_documents.append(get_document_from_csv(full_path))
                elif filename.lower().endswith(".xlsx"):
                    new_documents.append(get_document_from_xlsx(full_path))
                elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    new_documents.append(get_document_from_image(full_path))
                print(f"   - Processed: {filename}")
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")

        if new_documents:
             splitter = get_node_parser()
             nodes = splitter.get_nodes_from_documents(new_documents)

             if nodes:
                 print(f"📥 Inserting {len(nodes)} chunks into database...")
                 _index_instance.insert_nodes(nodes)
             else:
                 print("⚠️ No content chunks created from documents.")

    current_state = get_current_state(domain_path)
    with open(STATE_FILE, 'w') as f:
        json.dump(current_state, f)

    print("✅ Knowledge base updated!")

def rebuild_knowledge_base():
    print("⚠️  Initiating full knowledge base rebuild...")
    global _index_instance
    _index_instance = None
    gc.collect()
    time.sleep(0.5)
    db_path = settings.vector_store.path
    collection_name = settings.vector_store.collection_name

    try:
        db = chromadb.PersistentClient(path=db_path)
        try:
            db.delete_collection(name=collection_name)
            print(f"🗑️  Collection '{collection_name}' deleted via API.")
        except ValueError:
            pass
    except Exception as e:
        print(f"⚠️ API cleanup warning: {e}")

    if os.path.exists(db_path):
        print("🧹 Cleaning up storage artifacts...")
        for item in os.listdir(db_path):
            item_path = os.path.join(db_path, item)

            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"   - Removed artifact: {item}")

                elif os.path.isfile(item_path) and not item.endswith(".sqlite3"):
                     os.remove(item_path)

            except PermissionError:
                pass
            except Exception as e:
                print(f"   ⚠️ Could not remove {item}: {e}")

    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
            print(f"🗑️  Deleted state file.")
        except:
            pass

    print("🔄 Starting re-indexing process...")
    try:
        _index_instance = initialize_index()

        try:
            current_state = get_current_state(settings.domain.domain_path)
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump(current_state, f)
            print("💾 New state file saved.")
        except Exception as e:
            print(f"⚠️ Warning: Could not save state file: {e}")

    except Exception as e:
        print(f"❌ Critical error during indexing: {e}")

def initialize_index():
    db_path = settings.vector_store.path
    collection_name = settings.vector_store.collection_name
    db = chromadb.PersistentClient(path=db_path)
    hnsw_config = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:M": 64,
        "hnsw:search_ef": 200
    }
    chroma_collection = db.get_or_create_collection(name=collection_name, metadata=hnsw_config)
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

def initialize_memory():
    global _memory
    if _memory is None:
        _memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    return _memory

def reset_chat_history():
    global _memory
    if _memory:
        _memory.reset()

_index_instance = initialize_index()
_memory = None
sync_access_levels()

def get_response(query_text: str, file_filters: list[str] = []):
    if _index_instance is None:
        raise ValueError("Index is not initialized. Run rebuild or check startup.")

    if query_text.strip().lower() == "/reset":
        reset_chat_history()
        return "Chat history cleared."

    filters_list = [
        MetadataFilter(key="access_level", value=ACCESS_CONTROL_STATUS)
    ]

    if file_filters:
        print(f"🔍 Filtering chat by documents: {file_filters}")
        filters_list.append(
            MetadataFilter(
                key="file_name",
                value=file_filters,
                operator=FilterOperator.IN
            )
        )

    filters = MetadataFilters(
        filters=filters_list,
        condition=FilterCondition.AND
    )
    memory = initialize_memory()
    chat_engine = _index_instance.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        filters=filters,
        similarity_top_k=settings.vector_store.top_k,
        context_prompt=(
            "You are a helpful assistant capable of answering questions about the provided documents.\n"
            "Here are the relevant documents for the context:\n"
            "{context_str}\n"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
            "Keep your answers short, direct, and to the point. "
            "Do not use ASCII tables or markdown tables unless explicitly asked. "
            "Use markdown formatting for answers. "
            "CRITICAL INSTRUCTION: If the provided context does NOT contain the facts to answer the question, "
            "you MUST say 'I cannot answer this based on the available documents'. "
        ),
        verbose=False
    )
    response = chat_engine.chat(query_text)
    return response

if __name__ == "__main__":
    print(get_documents_access_control())
