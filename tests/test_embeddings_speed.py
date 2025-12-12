import os
import time
import asyncio
from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

NUM_DOCS = 1000
DOC_LENGTH_WORDS = 150
LOCAL_BATCH_SIZE = 8
OPENAI_BATCH_SIZE = 256

SAMPLE_TEXT = "This is a test sentence to simulate a document chunk. " * (DOC_LENGTH_WORDS // 10)
DOCUMENTS = [SAMPLE_TEXT for _ in range(NUM_DOCS)]
QUERY = "How to optimize vector search speed?"

def get_models() -> List[tuple[str, BaseEmbedding]]:
    models = []

    try:
        models.append((
            "Snowflake Arctic (Local)",
            HuggingFaceEmbedding(
                model_name="Snowflake/snowflake-arctic-embed-m-v2.0",
                trust_remote_code=True,
                model_kwargs={"attn_implementation": "sdpa"},
                embed_batch_size=LOCAL_BATCH_SIZE
            )
        ))
    except Exception as e: print(f"Skipping Snowflake: {e}")

    if os.getenv("OPENAI_API_KEY"):
        models.append((
            "OpenAI (text-embedding-3-large)",
            OpenAIEmbedding(
                model="text-embedding-3-large",
                dimensions=512,
                # Увеличиваем батч для API, чтобы слать меньше запросов
                embed_batch_size=OPENAI_BATCH_SIZE
            )
        ))
    else:
        print("⚠️ OPENAI_API_KEY not found.")

    try:
        models.append((
            "all-MiniLM-L12-v2",
            HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L12-v2",
                embed_batch_size=LOCAL_BATCH_SIZE
            )
        ))
    except Exception as e:
        print(f"Skipping MiniLM: {e}")

    try:
        models.append((
            "MongoDB (mdbr-leaf-ir)",
            HuggingFaceEmbedding(
                model_name="MongoDB/mdbr-leaf-ir",
                embed_batch_size=LOCAL_BATCH_SIZE
            )
        ))
    except Exception as e:
        print(f"Skipping MongoDB model: {e}")

    return models

async def benchmark_model(name, model):
    try:
        await model.aget_query_embedding("warmup")
        start_t = time.perf_counter()
        _ = await model.aget_query_embedding(QUERY)
        query_time_ms = (time.perf_counter() - start_t) * 1000
        start_t = time.perf_counter()
        _ = await model.aget_text_embedding_batch(DOCUMENTS)
        indexing_time_s = time.perf_counter() - start_t

        docs_per_sec = NUM_DOCS / indexing_time_s

        print(f"{name:<35} | {query_time_ms:6.1f} ms | {indexing_time_s:10.2f} s | {docs_per_sec:8.1f}")

    except Exception as e:
        print(f"{name:<35} | {'ERROR':<10} | {str(e)}")

async def run_async_benchmark():
    models = get_models()

    print(f"\n🚀 STARTING ASYNC SPEED BENCHMARK")
    print(f"Documents: {NUM_DOCS}")
    print(f"Local Batch: {LOCAL_BATCH_SIZE} | OpenAI Batch: {OPENAI_BATCH_SIZE}")
    print("-" * 80)
    print(f"{'Model Name':<35} | {'Query (ms)':<10} | {'Indexing (s)':<12} | {'Docs/Sec':<10}")
    print("-" * 80)

    for name, model in models:
        await benchmark_model(name, model)

    print("-" * 80)

if __name__ == "__main__":
    asyncio.run(run_async_benchmark())
