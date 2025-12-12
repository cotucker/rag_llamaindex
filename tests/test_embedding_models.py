import os
import numpy as np
from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

QUERY = "How do I reset my corporate password?"

DOCUMENTS = [
    # 🥇 PERFECT (Score 3) - Direct, actionable instructions for the user
    {"text": "IT Guide: To reset your corporate password, go to id.portal.com, click 'Forgot Password', and enter your 2FA code.", "label": "PERFECT", "score": 3},
    {"text": "Desktop Method: You can reset your password directly from the Windows login screen by clicking 'Reset Password / Unlock Account' below the login field.", "label": "PERFECT", "score": 3},
    {"text": "Phone Support: If you cannot access the web portal, call the IT Service Desk at ext. 5555 to request a manual password reset.", "label": "PERFECT", "score": 3},

    # 🥈 RELEVANT (Score 2) - Policy info, requirements, or related security context
    {"text": "Security Policy: Employees are required to change their passwords every 90 days. Contact HelpDesk if locked out.", "label": "RELEVANT", "score": 2},
    {"text": "Password Complexity: New passwords must be at least 12 characters long and contain an uppercase letter, a number, and a special symbol.", "label": "RELEVANT", "score": 2},
    {"text": "Account Lockout Policy: After 5 incorrect attempts, your corporate account will be locked for 30 minutes automatically.", "label": "RELEVANT", "score": 2},
    {"text": "MFA Setup: Changing your password will require you to re-authenticate your mobile device via the Authenticator app.", "label": "RELEVANT", "score": 2},

    # 🥉 WEAK (Score 1) - Tangentially related (onboarding, VPN, specific apps)
    {"text": "Onboarding: You will receive temporary credentials via email on your first day.", "label": "WEAK", "score": 1},
    {"text": "VPN Troubleshooting: If the VPN rejects your connection, ensure your domain password hasn't expired recently.", "label": "WEAK", "score": 1},
    {"text": "Jira Access: Your login for the project management tool is synced with your main corporate Active Directory account.", "label": "WEAK", "score": 1},

    # 😈 DISTRACTOR (Score 0) - TRAPS! High keyword overlap, wrong meaning
    {"text": "The password for the break room Wi-Fi is 'Coffee2024'. Do not share it with guests.", "label": "DISTRACTOR", "score": 0},
    {"text": "Server Maintenance: Resetting the production database requires root privileges and a sudo password.", "label": "DISTRACTOR", "score": 0},
    {"text": "Customer Support Guide: How to reset a *client's* password in the admin panel. Never ask the client for their old password.", "label": "DISTRACTOR", "score": 0},
    {"text": "Printer Troubleshooting: To factory reset the HP printer, hold the 'Cancel' and 'Power' buttons for 10 seconds.", "label": "DISTRACTOR", "score": 0},
    {"text": "Personal Devices: If you forgot your iPhone passcode, you will need to restore the device via iTunes.", "label": "DISTRACTOR", "score": 0},
    {"text": "Phishing Alert: IT will NEVER ask you to reset your password via a link sent in an SMS message. Report such attempts.", "label": "DISTRACTOR", "score": 0},
    {"text": "Meeting Room: The pin code to unlock the conference room iPad is 1234.", "label": "DISTRACTOR", "score": 0},

    # 🗑️ IRRELEVANT (Score 0) - Random noise
    {"text": "The annual company retreat will be held in Bali this year.", "label": "NOISE", "score": 0},
    {"text": "Python 3.12 introduces performance improvements to the interpreter.", "label": "NOISE", "score": 0},
    {"text": "Please remember to wash your dishes after using the kitchenette.", "label": "NOISE", "score": 0},
]

def get_models() -> List[tuple[str, BaseEmbedding]]:
    models = []

    try:
        models.append((
            "Snowflake Arctic (m-v2.0)",
            HuggingFaceEmbedding(
                model_name="Snowflake/snowflake-arctic-embed-m-v2.0",
                trust_remote_code=True,
                model_kwargs={"attn_implementation": "sdpa"}
            )
        ))
    except Exception as e: print(f"Skipping Snowflake: {e}")

    if os.getenv("OPENAI_API_KEY"):
        models.append((
            "OpenAI (text-embedding-3-large)",
            OpenAIEmbedding(model="text-embedding-3-large", dimensions=512)
        ))
    else:
        print("⚠️ OPENAI_API_KEY not found, skipping OpenAI model.")

    models.append((
        "all-MiniLM-L12-v2",
        HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")
    ))

    try:
        models.append((
            "MongoDB/mdbr-leaf-ir",
            HuggingFaceEmbedding(model_name="MongoDB/mdbr-leaf-ir")
        ))
    except Exception as e: print(f"Skipping MongoDB model: {e}")

    return models

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluate_model(name, model):
    print(f"\n{'='*60}")
    print(f"🤖 TESTING MODEL: {name}")
    print(f"{'='*60}")

    try:
        query_vec = model.get_query_embedding(QUERY)

        results = []
        for doc in DOCUMENTS:
            doc_vec = model.get_text_embedding(doc["text"])
            similarity = cosine_similarity(query_vec, doc_vec)
            results.append({**doc, "sim": similarity})

        results.sort(key=lambda x: x["sim"], reverse=True)
        score = 0
        max_score = 0
        ideal_scores = sorted([d["score"] for d in DOCUMENTS], reverse=True)
        print(f"Query: '{QUERY}'\n")
        print(f"{'Rank':<4} | {'Sim':<6} | {'Label':<10} | Text Snippet")
        print("-" * 60)

        for i, res in enumerate(results):
            rank = i + 1
            icon = "✅" if res["score"] > 0 else "❌"
            if res["label"] == "DISTRACTOR": icon = "😈"
            print(f"#{rank:<3} | {res['sim']:.4f} | {icon} {res['label']:<7} | {res['text'][:60]}...")
            discount = 1 / np.log2(rank + 1)
            score += res["score"] * discount

            if i < len(ideal_scores):
                max_score += ideal_scores[i] * discount

        ndcg = (score / max_score) * 100 if max_score > 0 else 0
        print(f"\n🏆 Model Quality (NDCxG): {ndcg:.2f}%")
        first_place = results[0]

        if first_place["score"] == 0:
            print("⚠️ WARNING: Model put irrelevant document at #1!")

    except Exception as e:
        print(f"❌ Error testing {name}: {e}")

if __name__ == "__main__":
    models = get_models()
    for name, model in models:
        evaluate_model(name, model)
