import os
import shutil
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cerebras import Cerebras
from dotenv import load_dotenv
from src.config import settings

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise ValueError("CEREBRAS_API_KEY environment variable is not set.")

Settings.llm = Cerebras(model="gpt-oss-120b", api_key=CEREBRAS_API_KEY)
embed_model = HuggingFaceEmbedding(model_name="Snowflake/snowflake-arctic-embed-m-v2.0", trust_remote_code=True)

def get_document_from_pdf(path_to_pdf: str) -> Document:
    import pymupdf
    doc = pymupdf.open(path_to_pdf)
    text = '\n'.join([page.get_text() for page in doc])
    return Document(text=text, metadata={"file_path": path_to_pdf, "file_name": os.path.basename(path_to_pdf)})

def get_document_from_txt(path_to_txt: str) -> Document:
    with open(path_to_txt, "r", encoding="utf-8") as f:
        text = f.read()
    return Document(text=text, metadata={"file_path": path_to_txt, "file_name": os.path.basename(path_to_txt)})

def get_documents(path: str):
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

    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=70,
        embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def test(qeury_text: str = "What did the author do growing up?"):
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # define embedding function
    embed_model = HuggingFaceEmbedding(model_name="Snowflake/snowflake-arctic-embed-m-v2.0", trust_remote_code=True)

    # load documents
    documents = get_documents("data/test_chunking")

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    # Query Data
    query_engine = index.as_query_engine(similarity_top_k=1)
    response = query_engine.query(qeury_text)
    print(str(response))
    print("Sources:")
    for node_with_score in response.source_nodes:
        score = node_with_score.score
        node = node_with_score.node
        print(f"Score: {score:.4f}")
        print(f"Text: {node.text}...")
        print(f"Metadata: {node.metadata}")
        print("-" * 30)


if __name__ == "__main__":
    test("The planet Mars")
