import os
import pandas as pd
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from pymongo import MongoClient
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Replace this with your actual MongoDB connection string
MONGO_URI = os.environ.get("MONGO_URI")
DATABASE_NAME = "fashion_matcher"
COLLECTION_NAME = "products"

# --- Model Setup (Using OpenAI's CLIP) ---
# CLIP is excellent for visual similarity search
MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"✅ CLIP Model loaded on device: {device}")
except Exception as e:
    print(f"❌ Error loading CLIP model or device: {e}")
    exit()

# --- Utility Function to Get Embedding ---
def get_image_embedding(image_url):
    """Fetches image from URL, processes it, and returns the embedding vector."""
    try:
        # 1. Fetch the image data from the URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes

        # 2. Load the image into PIL (in memory)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 3. Process the image for the CLIP model
        inputs = processor(images=image, return_tensors="pt").to(device)

        # 4. Generate the embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Convert the tensor to a list of floats (ready for MongoDB storage)
        embedding_list = image_features.cpu().numpy().flatten().tolist()
        return embedding_list

    except requests.exceptions.RequestException as e:
        # Handle network/URL errors (e.g., 404, timeout)
        print(f"   [Error] Could not fetch image from {image_url}: {e}")
        return None
    except Exception as e:
        # Handle other processing errors (e.g., corrupted image file)
        print(f"   [Error] Processing failed for {image_url}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print("✅ Connected to MongoDB.")

        # 2. Get products that DO NOT yet have an embedding
        # This allows you to re-run the script without re-processing everything.
        products_to_process = collection.find({"image_embedding": {"$exists": False}})
        total_count = collection.count_documents({"image_embedding": {"$exists": False}})

        print(f"Processing {total_count} products without embeddings...")
        
        # 3. Process and Update Loop
        for product in tqdm(products_to_process, total=total_count, desc="Generating Embeddings"):
            image_url = product.get("image_url")
            product_id = product.get("id")

            if image_url:
                embedding = get_image_embedding(image_url)
                
                if embedding:
                    # Update the document with the new embedding
                    collection.update_one(
                        {"id": product_id},
                        {"$set": {"image_embedding": embedding}}
                    )

        print("\n🎉 Step 3 Complete: All image embeddings have been generated and stored in MongoDB.")
        
    except Exception as e:
        print(f"\n🛑 A critical error occurred during MongoDB operation: {e}")
    finally:
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed.")