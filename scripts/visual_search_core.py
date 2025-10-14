import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import requests
import os
import sys
from sklearn.preprocessing import normalize
import faiss

# ======================================================
# --- Configuration (CRITICAL: Ensure URI is correct) ---
# ======================================================
MONGO_URI =os.environ.get("MONGO_URI")
DATABASE_NAME = "fashion_matcher"
COLLECTION_NAME = "products"
MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIMENSION = 512
INDEX_FILENAME = "faiss_index.bin"
PRODUCT_IDS_FILE = "product_ids.npy"

# ======================================================
# --- Global Components ---
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None
faiss_index = None
product_ids_map = None
db_collection = None


# ======================================================
# --- Utility: Environment Validation ---
# ======================================================
def validate_system_ready():
    """Ensures all global components are loaded and valid."""
    if model is None:
        return {"error": "Model not loaded."}
    if processor is None:
        return {"error": "Processor not loaded."}
    if faiss_index is None:
        return {"error": "FAISS index not loaded."}
    if product_ids_map is None or not isinstance(product_ids_map, np.ndarray):
        return {"error": "Product ID map missing or invalid."}
    if db_collection is None:
        return {"error": "MongoDB connection not established."}
    return None


# ======================================================
# --- Initialization ---
# ======================================================
def initialize_search_system():
    """Loads ML model, FAISS index, ID map, and establishes DB connection."""
    global model, processor, faiss_index, product_ids_map, db_collection

    print("\n--- Initializing Visual Search System ---")

    # 1. Load CLIP Model
    try:
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print(f"✅ CLIP Model loaded on device: {device}")
    except Exception as e:
        print(f"❌ Error loading CLIP model: {e}")
        sys.exit(1)

    # 2. Load FAISS Index and Product ID Map
    try:
        if not os.path.exists(INDEX_FILENAME) or not os.path.exists(PRODUCT_IDS_FILE):
            print(f"🛑 Error: Missing FAISS index or product ID map.")
            print("Please run build_faiss_index.py first.")
            sys.exit(1)

        faiss_index = faiss.read_index(INDEX_FILENAME)
        product_ids_map = np.load(PRODUCT_IDS_FILE)
        print(f"✅ FAISS Index loaded ({faiss_index.ntotal} vectors).")
        print(f"✅ Product ID Map loaded ({len(product_ids_map)} IDs).")
    except Exception as e:
        print(f"❌ Error loading FAISS index or ID map: {e}")
        sys.exit(1)

    # 3. Connect to MongoDB
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        db_collection = db[COLLECTION_NAME]
        print("✅ Connected to MongoDB.")
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        sys.exit(1)

    print("--- System Ready for Queries ---\n")


# ======================================================
# --- Query Embedding ---
# ======================================================
def get_query_embedding(image_url):
    """Fetches image from URL, processes it, and returns normalized embedding."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        query_vector = image_features.cpu().numpy().astype("float32")
        normalized_vector = normalize(query_vector, axis=1, copy=False)

        return normalized_vector

    except requests.exceptions.RequestException as e:
        print(f"[Error] Could not fetch query image from {image_url}: {e}")
        return None
    except Exception as e:
        print(f"[Error] Query image processing failed: {e}")
        return None


# ======================================================
# --- FAISS Search ---
# ======================================================
def search_similar_products(query_image_url: str, k: int = 10):
    """
    Performs the full visual search pipeline: encode → search → fetch metadata.
    """
    readiness_error = validate_system_ready()
    if readiness_error:
        return readiness_error

    # 1. Get Query Embedding
    query_vector = get_query_embedding(query_image_url)
    if query_vector is None:
        return {"error": "Failed to process the query image URL."}

    print(f"[Debug] Query vector shape: {query_vector.shape}, dtype: {query_vector.dtype}")

    query_vector_faiss_ready = np.array(query_vector, dtype=np.float32, copy=True, order="C")

    if query_vector_faiss_ready.ndim == 1:
        query_vector_faiss_ready = query_vector_faiss_ready.reshape(1, -1)

    if np.any(np.isnan(query_vector_faiss_ready)) or np.any(np.isinf(query_vector_faiss_ready)):
        print("[Error] Query vector contains NaN or Inf values.")
        return {"error": "Invalid query vector (NaN/Inf detected)."}

    # 2. Search in FAISS Index
    try:
        D, I = faiss_index.search(query_vector_faiss_ready, k)
    except Exception as e:
        print(f"[CRITICAL] FAISS search failed: {e}")
        return {"error": f"FAISS search error: {e}"}

    print(f"[Debug] FAISS Indices shape: {I.shape}, Distances shape: {D.shape}")

    faiss_indices = I[0]
    distances = D[0]
    result_product_ids_int = product_ids_map[faiss_indices].tolist()
    distances_list = distances.tolist()

    # --- 3. Fetch Metadata from MongoDB (Robust Double-Check) ---
    
    # Attempt 1: Query using Integer IDs
    product_cursor = db_collection.find({"id": {"$in": result_product_ids_int}})
    
    # Map the results, preferring the 'id' field, but falling back to '_id'
    product_data_map = {doc.get("id") or doc.get("_id"): doc for doc in product_cursor}

    # If the integer query returned no results, try the String IDs
    if not product_data_map and result_product_ids_int:
        print("[Debug] No results found with integer IDs. Re-querying with string IDs...")
        
        result_product_ids_str = [str(pid) for pid in result_product_ids_int]
        product_cursor_str = db_collection.find({"id": {"$in": result_product_ids_str}})
        
        # Rebuild map with string IDs (or fallback IDs)
        product_data_map = {doc.get("id") or doc.get("_id"): doc for doc in product_cursor_str}

    # If the map is still empty, the key might be wrong, or the data is missing.
    if not product_data_map:
        print("[Debug] CRITICAL: MongoDB query failed for both integer and string IDs on field 'id'. Please verify your MongoDB document structure (field name).")
        return []

    print(f"[Debug] Successfully retrieved {len(product_data_map)} products.")

    # 4. Combine results
    final_results = []
    # Loop over the original FAISS results and perform a dual-type lookup against the populated map
    for product_id_int, distance in zip(result_product_ids_int, distances_list):
        
        # Try lookup using the integer key (if Mongo doc ID is an integer)
        product = product_data_map.get(product_id_int)
        
        if not product:
            # Try lookup using the string key (if Mongo doc ID is a string)
            product = product_data_map.get(str(product_id_int))
            
        if product:
            similarity_score = 1.0 / (1.0 + distance)
            
            result_item = {
                "id": product.get("id", str(product.get("_id"))), 
                "name": product.get("productDisplayName"),
                "category": product.get("articleType"),
                "price": product.get("price", 0.0), 
                "imageUrl": product.get("image_url"),
                "similarityScore": float(similarity_score),
                "distance": float(distance),
            }
            final_results.append(result_item)

    final_results.sort(key=lambda x: x["similarityScore"], reverse=True)
    return final_results


# ======================================================
# --- Example Run (Using a high-quality test image) ---
# ======================================================
if __name__ == "__main__":
    initialize_search_system()

    # --- UPDATED TEST IMAGE URL (Red Printed T-Shirt) ---
    # This URL should ideally point to an image that is visually distinct 
    # and has similar items in your index.
    test_query_url = "http://assets.myntassets.com/v1/images/style/properties/4850873d0c417e6480a26059f83aac29_images.jpg"
    print(f"Searching for products similar to: {test_query_url}\n")

    try:
        results = search_similar_products(test_query_url, k=5)
        
        print("\n--- Top 5 Search Results ---")
        if isinstance(results, dict) and results.get("error"):
            print("Error:", results["error"])
        elif isinstance(results, list) and len(results) > 0:
            for i, item in enumerate(results, start=1):
                print(f"{i}. ID: {item['id']} | {item['name'][:40]}... | Score: {item['similarityScore']:.4f} | Category: {item['category']}")
        else:
            print("No results found.")
            
    except Exception as e:
        print(f"An unexpected error occurred during search execution: {e}")
