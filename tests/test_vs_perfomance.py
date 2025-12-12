import os
import shutil
import time
import numpy as np
import shutil
import uuid
from rich.console import Console
from rich.table import Table

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

VECTOR_DIM = 768
DATASET_SIZE = 100_000
QUERY_COUNT = 50
TEMP_DIR = "./bench_llamaindex_temp"

console = Console()

class MockEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text: str) -> list[float]:
        return np.random.rand(VECTOR_DIM).tolist()
    def _get_query_embedding(self, query: str) -> list[float]:
        return np.random.rand(VECTOR_DIM).tolist()
    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

mock_embed_model = MockEmbedding(embed_batch_size=100)
Settings.embed_model = mock_embed_model
Settings.llm = None

def generate_nodes(count):
    nodes = []
    for i in range(count):
        node = TextNode(
            text=f"This is chunk number {i}",
            id_=str(uuid.uuid4())
        )
        nodes.append(node)
    return nodes

def get_folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def cleanup():
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except:
            pass
    os.makedirs(TEMP_DIR, exist_ok=True)

def run_store_test(name, vector_store, nodes):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    start_time = time.perf_counter()
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=mock_embed_model,
        show_progress=True
    )
    indexing_time = time.perf_counter() - start_time
    retriever = index.as_retriever(similarity_top_k=5)
    start_time = time.perf_counter()
    for i in range(QUERY_COUNT):
        retriever.retrieve(f"test query {i}")
    query_time = (time.perf_counter() - start_time) / QUERY_COUNT
    return indexing_time, query_time

def run_benchmark():
    cleanup()
    console.print(f"[bold cyan]🚀 Starting LlamaIndex Benchmark: {DATASET_SIZE} nodes[/bold cyan]\n")
    with console.status("Generating mock nodes..."):
        nodes = generate_nodes(DATASET_SIZE)
    results = []
    console.print("Testing [yellow]ChromaDB[/yellow]...")
    try:
        path = os.path.join(TEMP_DIR, "chroma")
        if os.path.exists("./chroma_bench"):
            shutil.rmtree("./chroma_bench")
        chroma_client = chromadb.PersistentClient(path=path)
        chroma_collection = chroma_client.get_or_create_collection("bench")
        store = ChromaVectorStore(chroma_collection=chroma_collection)
        t_idx, t_query = run_store_test("Chroma", store, nodes)
        size = get_folder_size(path)
        results.append(["ChromaDB", f"{t_idx:.2f}s", f"{t_query*1000:.2f}ms", f"{size:.1f} MB"])
        del chroma_client
    except Exception as e:
        console.print(f"[red]Chroma Failed:[/red] {e}")
    console.print("Testing [yellow]LanceDB[/yellow]...")
    try:
        path = os.path.join(TEMP_DIR, "lancedb")
        store = LanceDBVectorStore(uri=path, table_name="bench")
        t_idx, t_query = run_store_test("LanceDB", store, nodes)
        size = get_folder_size(path)
        results.append(["LanceDB", f"{t_idx:.2f}s", f"{t_query*1000:.2f}ms", f"{size:.1f} MB"])
    except Exception as e:
        console.print(f"[red]LanceDB Failed:[/red] {e}")
    console.print("Testing [yellow]Qdrant[/yellow]...")
    try:
        path = os.path.join(TEMP_DIR, "qdrant")
        client = qdrant_client.QdrantClient(path=path)
        store = QdrantVectorStore(client=client, collection_name="bench")
        t_idx, t_query = run_store_test("Qdrant", store, nodes)
        size = get_folder_size(path)
        results.append(["Qdrant", f"{t_idx:.2f}s", f"{t_query*1000:.2f}ms", f"{size:.1f} MB"])
    except Exception as e:
        import traceback
        console.print(f"[red]Qdrant Failed:[/red] {e}")
        console.print(traceback.format_exc())
    table = Table(title=f"LlamaIndex Benchmark ({DATASET_SIZE} items)")
    table.add_column("Database", style="bold magenta")
    table.add_column("Indexing Time", justify="right")
    table.add_column("Query Latency", justify="right")
    table.add_column("Disk Size", justify="right")
    for row in results:
        table.add_row(*row)
    console.print("\n")
    console.print(table)
    cleanup()

if __name__ == "__main__":
    run_benchmark()
