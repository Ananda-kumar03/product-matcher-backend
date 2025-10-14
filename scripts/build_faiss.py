import torch
import numpy as np
import faiss
from pymongo import MongoClient
from tqdm import tqdm
import os
import sys
from sklearn.preprocessing import normalize # <-- CRITICAL IMPORT

# --- Configuration (Must match your previous script) ---
MONGO_URI = os.environ.get("MONGO_URI")
COLLECTION_NAME = "products"
EMBEDDING_DIMENSION = 512 
INDEX_FILENAME = "faiss_index.bin"
PRODUCT_IDS_FILE = "product_ids.npy" 

def build_and_save_index():
    """
    Connects to MongoDB, retrieves all embeddings, builds a FAISS index,
    and saves the index and a map of IDs to a local file.
    """
    print("--- Starting FAISS Index Construction ---")
    
    try:
        # 1. Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print("✅ Connected to MongoDB.")

        # 2. Retrieve all products that have an embedding
        cursor = collection.find(
            {"image_embedding": {"$exists": True}},
            {"id": 1, "image_embedding": 1, "_id": 0}
        )
        
        products = list(cursor)
        
        if not products:
            print("🛑 Error: No products with image_embedding found in the database.")
            client.close()
            return
            
        print(f"Retrieved {len(products)} products with embeddings.")

        # 3. Separate IDs and Embeddings
        product_ids = [p['id'] for p in products]
        embeddings = [p['image_embedding'] for p in products]

        # 4. Convert to NumPy array (FAISS requires a contiguous NumPy array)
        vector_matrix = np.array(embeddings, dtype='float32')

        # Basic Sanity Check
        if vector_matrix.shape[1] != EMBEDDING_DIMENSION:
            print(f"🛑 Error: Expected dimension {EMBEDDING_DIMENSION}, but got {vector_matrix.shape[1]}. Check your CLIP model/data.")
            client.close()
            return

        # 4.5. CRITICAL FIX: Normalize vectors before indexing (L2 norm)
        # This is essential for IndexFlatL2 to function as Cosine Similarity.
        print("✅ Normalizing vectors using L2 norm...")
        vector_matrix = normalize(vector_matrix, axis=1, copy=False)
        
        # 5. Build the FAISS Index
        print("Building FAISS Index (IndexFlatL2)...")
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        # Add the normalized vectors
        index.add(vector_matrix)
        
        print(f"Index built successfully. Total vectors indexed: {index.ntotal}")
        
        # 6. Save the Index and ID Map
        faiss.write_index(index, INDEX_FILENAME)
        
        # Save the map that links FAISS index position to the MongoDB product 'id'.
        np.save(PRODUCT_IDS_FILE, np.array(product_ids))
        
        print(f"🎉 Success! Index saved to {INDEX_FILENAME}")
        print(f"🎉 Success! ID map saved to {PRODUCT_IDS_FILE}")

    except Exception as e:
        print(f"\n🛑 A critical error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    build_and_save_index()