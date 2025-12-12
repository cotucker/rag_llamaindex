import random
import shutil
import os
import time
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import chromadb
import qdrant_client

TOTAL_DOCS = 10000
HARD_NEGATIVES_COUNT = 50
QUERY = "How do I reset my password?"

hnsw_config = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 400,
    "hnsw:M": 128,
    "hnsw:search_ef": 400,
    "hnsw:space": "cosine"
}

needles = [
    # 🥇 PERFECT
    {"text": "IT Support Guide: To reset your corporate password, visit id.portal.com and click 'Forgot Password'. You will need your 2FA token.", "label": "perfect", "score": 3},
    {"text": "Self-Service Portal: Users can reset their AD password via the web interface at https://password-reset.internal. if they are locked out.", "label": "perfect", "score": 3},

    # 🥈 RELEVANT
    {"text": "Security Policy 2024: Employees must change passwords every 90 days. If you forget it, contact HelpDesk for a temporary token.", "label": "relevant", "score": 2},
    {"text": "Access Control Standard: Password complexity requires 12 characters. Resets are handled automatically through the Identity Provider.", "label": "relevant", "score": 2},
    {"text": "New Employee Onboarding: Your initial login credentials are sent to your personal email. Please change them immediately upon first login.", "label": "relevant", "score": 2},

    # 🥉 WEAK
    {"text": "Troubleshooting: If you cannot log in, check your Caps Lock key first. If the issue persists, your account might be locked due to failed attempts.", "label": "weak", "score": 1},
    {"text": "HelpDesk Hours: The support team is available 24/7 for account lockouts and credential issues.", "label": "weak", "score": 1},
    {"text": "VPN Connection Errors: Often caused by expired passwords. Ensure your credentials are up to date before connecting.", "label": "weak", "score": 1},
    {"text": "Mobile Device Management: You will be prompted to re-enter your exchange password after a system update.", "label": "weak", "score": 1},
    {"text": "Legacy Systems: The mainframe uses a separate password from your SSO login. It does not sync with the web portal.", "label": "weak", "score": 1},
]

NEEDLES_COUNT = len(needles)

hard_negatives_templates = [
    "I forgot my password for the coffee machine app.",
    "Resetting the server requires admin privileges and a root password.",
    "The password for the Wi-Fi is written on the whiteboard.",
    "Do not write your password on sticky notes.",
    "Password sharing violation report #442.",
    "How to reset the factory settings on the printer.",
    "Login failed: Invalid username or password.",
    "The word 'password' is the most common password.",
    "System reset initiated due to critical error.",
    "My keyboard is broken, I cannot type my password.",
]

noise_topics = [
    "The weather in London is rainy today.",
    "Recipe for chocolate cake: add flour and sugar.",
    "Python 3.11 introduces better error messages.",
    "The stock market closed with a minor loss.",
    "History of Rome: Julius Caesar was assassinated.",
    "Mars rover Perseverance sends new photos.",
    "Office lunch menu: Pizza and Salad.",
    "Meeting notes: Discuss Q4 marketing strategy.",
]

def generate_dataset():
    data = []
    data.extend(needles)

    for _ in range(HARD_NEGATIVES_COUNT):
        text = random.choice(hard_negatives_templates) + f" [Log ID: {random.randint(1000, 9999)}]"
        data.append({"text": text, "label": "distractor", "score": 0})

    remaining = TOTAL_DOCS - len(data)
    for i in range(remaining):
        text = random.choice(noise_topics) + f" [Record {i}]"
        data.append({"text": text, "label": "noise", "score": 0})

    random.shuffle(data)
    return [Document(text=d["text"], metadata={"label": d["label"], "score": d["score"]}) for d in data]

def calculate_score(results):
    print(f"\n🔍 Top {len(results)} Results:")
    found_score = 0
    found_relevance = []

    for i, node in enumerate(results):
        relevance = node.metadata.get("score", 0)
        label = node.metadata.get("label", "noise")

        icon = "❓"
        if label == "perfect": icon = "🥇"
        elif label == "relevant": icon = "🥈"
        elif label == "weak": icon = "🥉"
        elif label == "distractor": icon = "😈"
        elif label == "noise": icon = "🗑️"

        print(f"  #{i+1:<2}: {icon} [{label.upper():<9}] (Score: {node.score:.4f}) - {node.text[:80]}...")
        found_score += relevance / (i + 1)
        found_relevance.append(relevance)

    ideal_scores = sorted([n["score"] for n in needles], reverse=True)
    max_possible = 0

    for i in range(len(results)):
        if i < len(ideal_scores):
            max_possible += ideal_scores[i] / (i + 1)

    if max_possible == 0: return 0

    return (found_score / max_possible) * 100

embed_model = HuggingFaceEmbedding(model_name="Snowflake/snowflake-arctic-embed-m-v2.0", trust_remote_code=True)
Settings.embed_model = embed_model
Settings.llm = None
documents = generate_dataset()

def run_test(db_type):
    print(f"\n{'='*40}")
    print(f"🚀 TESTING: {db_type.upper()}")
    print(f"{'='*40}")

    storage_context = None

    try:
        if db_type == "chroma":
            if os.path.exists("./chroma_bench"): shutil.rmtree("./chroma_bench")
            db = chromadb.PersistentClient(path="./chroma_bench")
            collection = db.get_or_create_collection("bench", metadata=hnsw_config)
            vector_store = ChromaVectorStore(chroma_collection=collection)

        elif db_type == "lancedb":
            if os.path.exists("./lancedb_bench"): shutil.rmtree("./lancedb_bench")
            vector_store = LanceDBVectorStore(uri="./lancedb_bench", table_name="vectors")

        elif db_type == "qdrant":
            path = "./qdrant_bench"
            if os.path.exists(path): shutil.rmtree(path)

            client = qdrant_client.QdrantClient(path=path) # Local persistence
            vector_store = QdrantVectorStore(client=client, collection_name="bench")

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        t_start = time.perf_counter()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        t_index = time.perf_counter() - t_start
        print(f"⏱️  Indexing Time: {t_index:.2f}s")
        t_start = time.perf_counter()
        retriever = index.as_retriever(similarity_top_k=10)
        results = retriever.retrieve(QUERY)
        t_search = time.perf_counter() - t_start
        print(f"⏱️  Search Time: {t_search*1000:.2f}ms")
        score = calculate_score(results)
        print(f"\n🏆 Quality Score: {score:.2f}%")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test("chroma")
    run_test("lancedb")
    run_test("qdrant")
