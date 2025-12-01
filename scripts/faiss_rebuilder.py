import os
import numpy as np
import faiss
from pymongo import MongoClient

# ---------------------------
# MongoDB Connection
# ---------------------------
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "fashion_matcher"
COLLECTION_NAME = "visualproducts"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# ---------------------------
# Load embeddings from MongoDB
# ---------------------------
print("📥 Loading embeddings from MongoDB...")

docs = list(col.find({}, {"_id": 1, "mnetv2_embedding_1280": 1}))

embeddings = []
product_ids = []

for d in docs:
    emb = d.get("mnetv2_embedding_1280")
    if emb:
        embeddings.append(np.array(emb, dtype="float32"))
        product_ids.append(str(d["_id"]))

embeddings = np.vstack(embeddings)
product_ids = np.array(product_ids)

print(f"🔢 Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

# ---------------------------
# Build FAISS index
# ---------------------------
dim = 1280
index = faiss.IndexFlatL2(dim)

print("⚙️ Adding vectors to FAISS index...")
index.add(embeddings)

# ---------------------------
# Save index + product ID mapping
# ---------------------------
faiss.write_index(index, "faiss_index.bin")
np.save("product_ids.npy", product_ids)

print("🎉 FAISS index saved as faiss_index.bin")
print("🎉 Product IDs saved as product_ids.npy")

