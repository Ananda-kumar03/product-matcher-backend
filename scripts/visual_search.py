# import os
# import sys
# import numpy as np
# import requests
# from io import BytesIO
# from PIL import Image
# from typing import List, Dict, Any

# # Required external libraries
# import faiss
# from pymongo import MongoClient
# from google import genai
# from google.genai import types 
# from google.genai.errors import APIError

# # ======================================================
# # --- Configuration (CRITICAL) ---
# # ======================================================
# # Use your actual MongoDB connection string from the environment or hardcode it (not recommended)
# MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://anandkumarvijay11:pGoJIWwIn26Yrzhv@cluster0.qja0w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# # !!! MANDATORY ACTION: REPLACE THIS PLACEHOLDER WITH YOUR VALID GEMINI API KEY !!!
# GEMINI_API_KEY = "AIzaSyAbE7ckmkEjPbC9rGZ8H8aYC8kfgaHTfLE" 
# # =================================================================================

# DATABASE_NAME = "fashion_matcher"
# COLLECTION_NAME = "visualproducts" # Collection containing product metadata (IDs must match FAISS)

# # --- GEMINI MODEL CONFIG ---
# EMBEDDING_MODEL = "gemini-embedding-001"
# EMBEDDING_DIMENSION = 3072

# # --- FAISS FILE CONFIG (Must exist locally) ---
# INDEX_FILENAME = "gemini_faiss_index_3072.bin"
# PRODUCT_IDS_FILE = "gemini_product_ids_3072.npy"

# # --- TEST QUERY IMAGE ---
# TEST_QUERY_URL = "http://assets.myntassets.com/v1/images/style/properties/4850873d0c417e6480a26059f83aac29_images.jpg"


# # ======================================================
# # --- Global Components ---
# # ======================================================
# gemini_client = None
# faiss_index = None
# product_ids_map = None
# db_collection = None


# # ======================================================
# # --- Utility: System Validation & Key Handling ---
# # ======================================================
# def validate_system_ready():
#     """Ensures all global components are loaded and valid before querying."""
#     if gemini_client is None:
#         return {"error": "Gemini client not initialized. Please provide a valid API Key."}
#     if faiss_index is None:
#         return {"error": "FAISS index not loaded. Check index file path."}
#     if product_ids_map is None or not isinstance(product_ids_map, np.ndarray):
#         return {"error": "Product ID map missing or invalid. Check NPY file."}
#     return None

# def get_gemini_api_key() -> str:
#     """Retrieves the hardcoded Gemini API key and validates it is not the placeholder."""
#     if GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
#         raise ValueError(
#             "Gemini API Key is missing. Please replace 'YOUR_ACTUAL_GEMINI_API_KEY_HERE' "
#             "with your valid API key in the GEMINI_API_KEY variable."
#         )
#     return GEMINI_API_KEY


# # ======================================================
# # --- Initialization ---
# # ======================================================
# def initialize_search_system():
#     """Loads FAISS index, ID map, establishes DB connection, and initializes Gemini client."""
#     global gemini_client, faiss_index, product_ids_map, db_collection

#     print("\n--- Initializing Visual Search System (Gemini + FAISS + Mongo) ---")
    
#     # 1. Initialize Gemini Client (Key is now explicitly hardcoded)
#     try:
#         # The key is retrieved from the hardcoded GEMINI_API_KEY variable
#         gemini_api_key = get_gemini_api_key() 
#         gemini_client = genai.Client(api_key=gemini_api_key)
#         print("✅ Gemini Client Initialized.")
#     except ValueError as e:
#         print(f"❌ Error initializing Gemini client: {e}")
#         # Stop execution if API key is missing
#         sys.exit(1)
#     except Exception as e:
#         print(f"❌ Unexpected error initializing Gemini client. Check the key's validity: {e}")
#         sys.exit(1)


#     # 2. Load FAISS Index and Product ID Map
#     try:
#         if not os.path.exists(INDEX_FILENAME) or not os.path.exists(PRODUCT_IDS_FILE):
#             print(f"🛑 Error: Missing FAISS index ({INDEX_FILENAME}) or product ID map ({PRODUCT_IDS_FILE}).")
#             sys.exit(1)

#         faiss_index = faiss.read_index(INDEX_FILENAME)
#         product_ids_map = np.load(PRODUCT_IDS_FILE)
#         print(f"✅ FAISS Index loaded ({faiss_index.ntotal} vectors, {EMBEDDING_DIMENSION}D).")
#         print(f"✅ Product ID Map loaded ({len(product_ids_map)} IDs).")
#     except Exception as e:
#         print(f"❌ Error loading FAISS index or ID map: {e}")
#         sys.exit(1)

#     # 3. Connect to MongoDB
#     try:
#         if MONGO_URI == "mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority":
#              print("⚠️ Warning: MONGO_URI is using a default placeholder. Please update your MONGO_URI.")
             
#         client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
#         client.admin.command('ping') 
#         db = client[DATABASE_NAME]
#         db_collection = db[COLLECTION_NAME]
#         print("✅ Connected to MongoDB.")
#     except Exception as e:
#         print(f"❌ Error connecting to MongoDB. Search results will be raw FAISS IDs only: {e}")
#         db_collection = None 

#     print("--- System Ready for Queries ---\n")


# # ======================================================
# # --- Query Embedding (Gemini) ---
# # ======================================================
# def get_query_embedding(image_url: str) -> np.ndarray | None:
#     """
#     Fetches image from URL, processes it, and returns the 3072D normalized 
#     embedding as a (1, 3072) float32 NumPy array for FAISS.
#     """
#     print(f"Generating 3072D embedding for: {image_url}")
#     try:
#         # 1. Download Image
#         response = requests.get(image_url, timeout=10)
#         response.raise_for_status()
#         image = Image.open(BytesIO(response.content))
        
#         # Prepare image bytes for API
#         img_byte_arr = BytesIO()
#         if image.mode in ('RGBA', 'P'):
#             image = image.convert('RGB')
#         image.save(img_byte_arr, format='JPEG') 
#         img_bytes = img_byte_arr.getvalue()

#         # 2. Construct API Content
#         text_part = types.Part(text="Embed this image for visual similarity search in a product catalog.")
#         image_part = types.Part(inline_data={"mime_type": "image/jpeg", "data": img_bytes})
        
#         content_block = types.Content(role="user", parts=[text_part, image_part])
        
#         # 3. Call Gemini API
#         # The client object (gemini_client) was initialized with the hardcoded key.
#         api_response = gemini_client.models.embed_content(
#             model=EMBEDDING_MODEL,
#             contents=[content_block]
#         )
        
#         if api_response and api_response.embeddings:
#             embedding_list = api_response.embeddings[0].values
#             query_vector = np.array([embedding_list], dtype=np.float32)
            
#             # L2 Normalize the vector
#             norm = np.linalg.norm(query_vector)
#             if norm != 0:
#                 query_vector /= norm 
            
#             print(f"✅ Embedding generated and normalized. Shape: {query_vector.shape}")
#             return query_vector
#         else:
#             raise APIError("Embedding API returned no content.")
            
#     except requests.exceptions.RequestException as e:
#         print(f"[Error] Could not fetch query image from {image_url}: {e}")
#     except APIError as e:
#         print(f"[Error] Gemini API Error during embedding generation: {e}")
#     except Exception as e:
#         print(f"[Error] Query image processing failed: {e}")
#     return None


# # ======================================================
# # --- FAISS and MongoDB Search ---
# # ======================================================
# def search_similar_products(query_image_url: str, k: int = 5) -> List[Dict[str, Any]] | Dict[str, str]:
#     """
#     Performs the full visual search pipeline: encode → search → fetch metadata.
#     """
#     readiness_error = validate_system_ready()
#     if readiness_error:
#         return readiness_error

#     # 1. Get Query Embedding
#     query_vector = get_query_embedding(query_image_url)
#     if query_vector is None:
#         return {"error": "Failed to generate query embedding."}

#     # 2. Search in FAISS Index
#     try:
#         D, I = faiss_index.search(query_vector, k)
#     except Exception as e:
#         print(f"[CRITICAL] FAISS search failed: {e}")
#         return {"error": f"FAISS search error: {e}"}

#     faiss_indices = I[0]
#     distances = D[0]
#     result_product_ids = product_ids_map[faiss_indices].tolist()
#     distances_list = distances.tolist()
    
#     if db_collection is None:
#         # MongoDB is unavailable, return raw FAISS results
#         print("Returning raw FAISS results due to MongoDB connection failure.")
#         return [{"id": pid, "distance": d} for pid, d in zip(result_product_ids, distances_list)]
        

#     # 3. Fetch Metadata from MongoDB 
    
#     int_ids = [pid for pid in result_product_ids if isinstance(pid, (int, np.integer))]
#     str_ids = [str(pid) for pid in result_product_ids]
    
#     product_cursor = db_collection.find({
#         "$or": [
#             {"id": {"$in": int_ids}},
#             {"id": {"$in": str_ids}}
#         ]
#     })

#     product_data_map = {}
#     for doc in product_cursor:
#         key = doc.get("id") if doc.get("id") is not None else str(doc.get("_id"))
#         product_data_map[key] = doc

#     print(f"[Debug] Successfully retrieved {len(product_data_map)} products from MongoDB.")

#     # 4. Combine results
#     final_results = []
    
#     for product_id_int, distance in zip(result_product_ids, distances_list):
        
#         product = product_data_map.get(product_id_int) # Try int key
#         if not product:
#             product = product_data_map.get(str(product_id_int)) # Try string key
            
#         if product:
#             # Score conversion: 1 / (1 + distance)
#             similarity_score = 1.0 / (1.0 + distance)
            
#             result_item = {
#                 "id": product.get("id", str(product.get("_id"))), 
#                 "name": product.get("productDisplayName", "N/A"),
#                 "category": product.get("articleType", "N/A"),
#                 "price": product.get("price", 0.0), 
#                 "imageUrl": product.get("image_url", "N/A"),
#                 "similarityScore": float(similarity_score),
#                 "distance": float(distance),
#             }
#             final_results.append(result_item)
#         else:
#             print(f"[Warning] Product ID {product_id_int} found in FAISS but not in MongoDB. Skipping.")

#     final_results.sort(key=lambda x: x["similarityScore"], reverse=True)
#     return final_results


# # ======================================================
# # --- Example Execution ---
# # ======================================================
# if __name__ == "__main__":
#     initialize_search_system()

#     print(f"Searching for products similar to: {TEST_QUERY_URL}\n")

#     try:
#         results = search_similar_products(TEST_QUERY_URL, k=5)
        
#         print("\n--- Top 5 Search Results ---")
#         if isinstance(results, dict) and results.get("error"):
#             print("Error:", results["error"])
#         elif isinstance(results, list) and len(results) > 0 and 'distance' in results[0]:
#             for i, item in enumerate(results, start=1):
#                 name = item.get('name', 'N/A (Metadata Missing)')
#                 category = item.get('category', 'N/A')
                
#                 print(f"{i}. ID: {item['id']} | Name: {name[:40]}... | Score: {item['similarityScore']:.4f} | Distance: {item['distance']:.4f} | Category: {category}")
#         else:
#             print("No results found or metadata fetch failed.")
            
#     except Exception as e:
#         print(f"An unexpected error occurred during search execution: {e}")


# import os
# import faiss
# import pickle
# import numpy as np
# from pymongo import MongoClient
# from tqdm import tqdm
# from google import genai
# from google.genai import types
# import requests

# # ------------------ CONFIG ------------------
# GEMINI_API_KEY = "AIzaSyC-EWjNT0H6-o2VMLBIOaJekVrLNZ9DUVY"  # <<< PUT YOUR API KEY
# DB_URI = "mongodb+srv://anandkumarvijay11:pGoJIWwIn26Yrzhv@cluster0.qja0w.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# DB_NAME = "fashion_matcher"
# COLLECTION = "visualproducts"

# INDEX_FILE = "faiss_index.bin"
# ID_FILE = "product_ids.pkl"

# EMBED_MODEL = "gemini-embedding-001"  # Gemini multimodal embedding
# EXPECTED_DIM = 3072

# # ------------------ INITIALIZE ------------------
# client = genai.Client(api_key=GEMINI_API_KEY)
# mongo = MongoClient(DB_URI)
# db = mongo[DB_NAME]
# col = db[COLLECTION]

# # ------------------ GEMINI EMBEDDING ------------------
# def get_image_embedding(image_url: str):
#     """Return Gemini embedding for an image URL."""
#     try:
#         image_bytes = requests.get(image_url, timeout=10).content
#     except Exception as e:
#         print(f"❌ Failed to load image {image_url}: {e}")
#         return None

#     try:
#         text_part = types.Part(text="Product image")
#         image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
#         content_obj = types.Content(parts=[text_part, image_part])

#         response = client.models.embed_content(
#             model=EMBED_MODEL,
#             contents=[content_obj]
#         )

#         return response.embeddings[0].values
#     except Exception as e:
#         print(f"❌ Gemini embedding failed for {image_url}: {e}")
#         return None

# # ------------------ BUILD FAISS INDEX ------------------
# def build_faiss_index():
#     print("🔍 Loading embeddings from MongoDB...")
#     products = list(col.find({"gemini_embedding": {"$exists": True}}))
#     print(f"✅ Found {len(products)} products with embeddings.")

#     embeddings, product_ids = [], []

#     for p in tqdm(products, desc="Preparing embeddings"):
#         emb = p.get("gemini_embedding")
#         if emb:
#             embeddings.append(emb)
#             product_ids.append(str(p["_id"]))

#     embeddings = np.array(embeddings, dtype=np.float32)
#     faiss.normalize_L2(embeddings)

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)

#     faiss.write_index(index, INDEX_FILE)
#     with open(ID_FILE, "wb") as f:
#         pickle.dump(product_ids, f)

#     print(f"✅ FAISS index built and saved ({len(product_ids)} products).")

# # ------------------ LOAD FAISS INDEX ------------------
# def load_faiss_index():
#     if not os.path.exists(INDEX_FILE) or not os.path.exists(ID_FILE):
#         build_faiss_index()

#     index = faiss.read_index(INDEX_FILE)
#     with open(ID_FILE, "rb") as f:
#         product_ids = pickle.load(f)

#     print(f"✅ Loaded FAISS index ({len(product_ids)} products).")
#     return index, product_ids

# # ------------------ SEARCH ------------------
# def search_similar_products(query_url, top_k=5):
#     print(f"\n🔍 Generating embedding for query image: {query_url}")
#     query_emb = get_image_embedding(query_url)
#     if query_emb is None:
#         print("❌ Failed to generate embedding for query.")
#         return

#     query_emb = np.array(query_emb, dtype=np.float32).reshape(1, -1)
#     faiss.normalize_L2(query_emb)

#     index, product_ids = load_faiss_index()
#     D, I = index.search(query_emb, top_k)

#     print(f"\n🏆 Top {top_k} similar products:")
#     for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
#         pid = product_ids[idx]
#         prod = col.find_one({"_id": pid})
#         print(f"{rank}. Product ID: {pid}, Similarity: {score:.4f}, Image URL: {prod.get('image_url')}")

# # ------------------ USAGE ------------------
# if __name__ == "__main__":
#     query_image_url = "http://assets.myntassets.com/v1/images/style/properties/7a5b82d1372a7a5c6de67ae7a314fd91_images.jpg"
#     search_similar_products(query_image_url, top_k=5)


import os
import faiss
import numpy as np
import requests
import io
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from pymongo import MongoClient
from bson.objectid import ObjectId
from PIL import Image

# ------------------ CONFIG ------------------
DB_URI = os.environ.get("MONGO_URI")
COLLECTION = "visualproducts"

# The names of the FAISS index and product ID mapping file you saved
INDEX_FILE = "faiss_index.bin" 
ID_FILE = "product_ids.npy"

# MobileNetV2 embedding dimension
EXPECTED_DIM = 1280 

# ------------------ INITIALIZE MONGODB ------------------
mongo = MongoClient(DB_URI)
db = mongo[DB_NAME]
col = db[COLLECTION]

# -----------------------------------------
# Load MobileNetV2 Model & Preprocessing Transform
# NOTE: This must match the model used for initial embedding
# -----------------------------------------
def load_mobilenetv2():
    """Loads MobileNetV2 model for feature extraction."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    
    # Remove classifier to get the feature vector
    model.classifier = nn.Identity()
    
    full_model = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    return full_model

mobilenet_model = load_mobilenetv2()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------------
# Helper: Download image
# -----------------------------------------
def download_image(url):
    """Downloads image bytes from a URL and returns a PIL Image."""
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            print(f"❌ HTTP {res.status_code} on download: {url}")
            return None
        # Use io.BytesIO to open the image directly from memory
        return Image.open(io.BytesIO(res.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return None

# -----------------------------------------
# Helper: Get MobileNetV2 embedding
# NOTE: This is the same logic as in your embedding script
# -----------------------------------------
def get_mobilenet_embedding(img):
    """Generates the 1280-dim embedding vector using MobileNetV2."""
    try:
        # 1. Apply transforms (resize, ToTensor, normalize)
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            # 2. Pass through the model
            emb = mobilenet_model(img_t).squeeze().numpy()

        # 3. Check dimension
        if len(emb) != EXPECTED_DIM:
             print(f"❌ Embedding had wrong dimension ({len(emb)}). Expected {EXPECTED_DIM}")
             return None
        
        return emb.astype(np.float32) # Ensure it's the correct FAISS type
    except Exception as e:
        print("❌ Embedding failed:", e)
        return None

# ------------------ LOAD FAISS INDEX ------------------
def load_faiss_index():
    """Loads the FAISS index and the corresponding product ID mapping."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(ID_FILE):
        print(f"❌ Required files not found: {INDEX_FILE} or {ID_FILE}. Run the index generation script first.")
        raise FileNotFoundError
    
    index = faiss.read_index(INDEX_FILE)
    product_ids = np.load(ID_FILE, allow_pickle=True)

    print(f"✅ Loaded FAISS index (Size: {index.ntotal}, Dim: {index.d}).")
    return index, product_ids

# ------------------ SEARCH ------------------
def search_similar_products(query_url, top_k=5):
    print(f"\n🔍 Processing query image: {query_url}")
    
    # 1. Download and embed the query image (omitted for brevity)
    img = download_image(query_url)
    if img is None:
        print("❌ Query image processing failed.")
        return

    query_emb = get_mobilenet_embedding(img)
    if query_emb is None:
        print("❌ Query embedding generation failed.")
        return

    query_emb = query_emb.reshape(1, -1)

    try:
        index, product_ids = load_faiss_index()
    except FileNotFoundError:
        return
        
    D, I = index.search(query_emb, top_k)

    print(f"\n🏆 Top {top_k} similar products (using L2 distance):")
    
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx >= len(product_ids):
             print(f"{rank}. Index out of bounds (Index: {idx}).")
             continue
        
        pid_str = product_ids[idx] # This is the product ID string
        
        try:
            # 2. Convert the string ID to a MongoDB ObjectId
            pid_obj = ObjectId(pid_str) 
        except Exception as e:
            print(f"⚠️ Skipping product {pid_str}: Invalid ObjectId format. Error: {e}")
            continue # Skip to the next result if the ID is malformed
            
        # 3. Retrieve product details from MongoDB using the ObjectId
        prod = col.find_one({"_id": pid_obj})
        
        # 4. Check if a document was found before accessing attributes
        if prod:
            # L2 Distance: LOWER score means HIGHER similarity.
            print(f"{rank}. Product ID: {pid_str}, L2 Distance: {score:.4f}, Image URL: {prod.get('image_url')}")
        else:
            print(f"{rank}. Product ID: {pid_str}, L2 Distance: {score:.4f}, Status: ❌ Document not found in MongoDB.")

            
# ------------------ USAGE ------------------
if __name__ == "__main__":
    # Test URL - Use a URL that looks like a product image
    query_image_url = "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8c2hvZXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=500"
    search_similar_products(query_image_url, top_k=5)


