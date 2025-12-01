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


