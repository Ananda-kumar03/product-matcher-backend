# import os
# from flask import Flask, request, jsonify
# from pymongo import MongoClient
# import bcrypt
# from datetime import datetime
# from bson.objectid import ObjectId
# from flask_cors import CORS
# # --- New Imports for Visual Search ---
# import torch
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import requests
# from sklearn.preprocessing import normalize
# import faiss
# import random  # For random price generation
# # -------------------------------------

# # --- Configuration and Setup ---

# # Initialize Flask App
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # --- MongoDB Setup ---
# MONGO_URI = os.environ.get("MONGO_URI")
# DB_NAME = 'fashion_matcher'
# USERS_COLLECTION_NAME = 'users'
# PRODUCTS_COLLECTION_NAME = 'products'

# # --- Product and Search Configuration ---
# PRODUCTS_PER_PAGE = 20
# EMBEDDING_DIMENSION = 512

# # --- File Paths ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# INDEX_FILENAME = os.path.join(BASE_DIR, "faiss_index.bin")
# PRODUCT_IDS_FILE = os.path.join(BASE_DIR, "product_ids.npy")
# MODEL_NAME = "openai/clip-vit-base-patch32"

# # Global components
# users_collection = None
# products_collection = None
# search_model = None
# search_processor = None
# faiss_index = None
# product_ids_map = None
# search_device = "cuda" if torch.cuda.is_available() else "cpu"

# # --- Database Connection (No model loading yet) ---
# try:
#     client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
#     client.admin.command('ping')

#     db = client[DB_NAME]
#     users_collection = db[USERS_COLLECTION_NAME]
#     products_collection = db[PRODUCTS_COLLECTION_NAME]
#     users_collection.create_index("email", unique=True)

#     print("✅ MongoDB: Connected to users and products collections. 'email' index ensured.")
#     print("⚙️ Visual Search components will load lazily on first use (Render safe).")

# except Exception as e:
#     print(f"❌ CRITICAL ERROR: Failed to connect to MongoDB. Error: {e}")

# # --- Utility Functions (Existing) ---
# def validate_data(data, required_fields):
#     if not data:
#         return False, "Missing JSON data in request body."
#     for field in required_fields:
#         if field not in data or not str(data[field]).strip():
#             return False, f"Missing or empty required field: {field}."
#     return True, None

# # --- Random Price Generator ---
# def generate_random_price():
#     """Generates a random price between 500 and 5000 (increments of 10)."""
#     return float(random.randint(50, 500) * 10)

# # --- Lazy Loading Helpers ---
# def ensure_clip_model_loaded():
#     """Lazily loads CLIP model and processor only once."""
#     global search_model, search_processor, search_device
#     if search_model is None or search_processor is None:
#         print("⚙️ Loading CLIP model for the first time...")
#         from transformers import CLIPProcessor, CLIPModel
#         search_model = CLIPModel.from_pretrained(MODEL_NAME).to(search_device)
#         search_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
#         print("✅ CLIP model loaded successfully.")

# def ensure_faiss_index_loaded():
#     """Lazily loads FAISS index and product ID map."""
#     global faiss_index, product_ids_map
#     if faiss_index is None or product_ids_map is None:
#         print("⚙️ Loading FAISS index for the first time...")
#         if os.path.exists(INDEX_FILENAME) and os.path.exists(PRODUCT_IDS_FILE):
#             faiss_index = faiss.read_index(INDEX_FILENAME)
#             product_ids_map = np.load(PRODUCT_IDS_FILE)
#             print(f"✅ FAISS index loaded successfully with {faiss_index.ntotal} vectors.")
#         else:
#             print("⚠️ FAISS index or product ID map not found on server.")
#             raise FileNotFoundError("Index or ID map missing.")

# # --- Visual Search Logic ---
# def get_query_embedding(image_source, is_file=False):
#     """
#     Fetches image (from URL or file), processes it, and returns normalized embedding.
#     Lazily loads CLIP model + processor only once.
#     """
#     ensure_clip_model_loaded()

#     try:
#         if is_file:
#             image = Image.open(image_source).convert("RGB")
#         else:
#             response = requests.get(image_source, timeout=10)
#             response.raise_for_status()
#             image = Image.open(BytesIO(response.content)).convert("RGB")

#         inputs = search_processor(images=image, return_tensors="pt").to(search_device)

#         with torch.no_grad():
#             image_features = search_model.get_image_features(**inputs)

#         query_vector = image_features.cpu().numpy().astype("float32")
#         normalized_vector = normalize(query_vector, axis=1, copy=False)
#         return normalized_vector

#     except Exception as e:
#         print(f"[Error] Query image processing failed: {e}")
#         return None

# def search_similar_products(query_vector, k: int = 20):
#     """Performs FAISS search and fetches metadata. Lazily loads FAISS if needed."""
#     ensure_faiss_index_loaded()
#     global faiss_index, product_ids_map, products_collection

#     query_vector_faiss_ready = np.array(query_vector, dtype=np.float32, copy=True, order="C").reshape(1, -1)
#     if np.any(np.isnan(query_vector_faiss_ready)):
#         return {"error": "Invalid query vector (NaN/Inf detected)."}

#     D, I = faiss_index.search(query_vector_faiss_ready, k)
#     faiss_indices = I[0]
#     distances = D[0]

#     result_product_ids_int = product_ids_map[faiss_indices].tolist()
#     distances_list = distances.tolist()

#     product_cursor = products_collection.find({"id": {"$in": result_product_ids_int}})
#     product_data_map = {doc.get("id") or doc.get("_id"): doc for doc in product_cursor}

#     if not product_data_map and result_product_ids_int:
#         result_product_ids_str = [str(pid) for pid in result_product_ids_int]
#         product_cursor_str = products_collection.find({"id": {"$in": result_product_ids_str}})
#         product_data_map = {doc.get("id") or doc.get("_id"): doc for doc in product_cursor_str}

#     final_results = []
#     for product_id_int, distance in zip(result_product_ids_int, distances_list):
#         product = product_data_map.get(product_id_int) or product_data_map.get(str(product_id_int))
#         if product:
#             similarity_score = 1.0 / (1.0 + distance)
#             price = generate_random_price()
#             final_results.append({
#                 "id": str(product.get("id", product.get("_id"))),
#                 "name": product.get("productDisplayName", "Unknown Product"),
#                 "category": product.get("articleType", "N/A"),
#                 "price": price,
#                 "imageUrl": product.get("image_url"),
#                 "similarityScore": float(similarity_score),
#             })

#     final_results.sort(key=lambda x: x["similarityScore"], reverse=True)
#     return final_results

# # --- User Signup ---
# @app.route('/api/signup', methods=['POST'])
# def signup():
#     if users_collection is None:
#         return jsonify({'error': 'Database connection failed. Check server logs.'}), 503

#     data = request.get_json()
#     is_valid, error = validate_data(data, ['username', 'email', 'password'])
#     if not is_valid:
#         return jsonify({'error': error}), 400

#     username = data['username'].strip()
#     email = data['email'].strip().lower()
#     password = data['password']

#     if users_collection.find_one({'email': email}):
#         return jsonify({'error': 'Email address already registered. Please try logging in.'}), 409

#     try:
#         password_bytes = password.encode('utf-8')
#         hashed_password = bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode('utf-8')
#     except Exception:
#         return jsonify({'error': 'Internal server error during password processing.'}), 500

#     new_user = {
#         'username': username,
#         'email': email,
#         'password_hash': hashed_password,
#         'created_at': datetime.utcnow()
#     }

#     try:
#         users_collection.insert_one(new_user)
#         return jsonify({'message': 'User created successfully.'}), 201
#     except Exception:
#         return jsonify({'error': 'Database error during account creation.'}), 500

# # --- User Login ---
# @app.route('/api/login', methods=['POST'])
# def login():
#     if users_collection is None:
#         return jsonify({'error': 'Database connection failed. Check server logs.'}), 503

#     data = request.get_json()
#     is_valid, error = validate_data(data, ['email', 'password'])
#     if not is_valid:
#         return jsonify({'error': error}), 400

#     email = data['email'].strip().lower()
#     password = data['password']
#     user = users_collection.find_one({'email': email})

#     if not user:
#         return jsonify({'error': 'Invalid email or password.'}), 401

#     try:
#         stored_hash_bytes = user['password_hash'].encode('utf-8')
#         password_bytes = password.encode('utf-8')

#         if bcrypt.checkpw(password_bytes, stored_hash_bytes):
#             user_id = str(user['_id'])
#             simulated_token = f"jwt_user_{user_id}_{int(datetime.utcnow().timestamp())}"
#             return jsonify({
#                 'message': 'Login successful.',
#                 'user_id': user_id,
#                 'username': user.get('username', 'User'),
#                 'token': simulated_token
#             }), 200
#         else:
#             return jsonify({'error': 'Invalid email or password.'}), 401

#     except Exception:
#         return jsonify({'error': 'Internal server error during login verification.'}), 500

# # --- Product Routes ---
# @app.route('/api/products', methods=['GET'])
# def get_products():
#     if products_collection is None:
#         return jsonify({'error': 'Product database connection failed.'}), 503

#     try:
#         page = int(request.args.get('page', 1))
#         if page < 1:
#             page = 1
#     except ValueError:
#         page = 1

#     skip_count = (page - 1) * PRODUCTS_PER_PAGE
#     try:
#         products_cursor = products_collection.find(
#             {},
#             {"id": 1, "productDisplayName": 1, "articleType": 1, "price": 1, "image_url": 1}
#         ).sort("id", 1).skip(skip_count).limit(PRODUCTS_PER_PAGE)

#         products_list = []
#         for doc in products_cursor:
#             price = generate_random_price()
#             products_list.append({
#                 "id": str(doc.get("id", doc.get("_id"))),
#                 "name": doc.get("productDisplayName", "N/A"),
#                 "category": doc.get("articleType", "N/A"),
#                 "price": price,
#                 "imageUrl": doc.get("image_url", "https://placehold.co/150x200/cccccc/000000?text=No+Image"),
#                 "similarityScore": 1.0
#             })

#         total_products = products_collection.count_documents({})
#         total_pages = (total_products + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE

#         return jsonify({
#             'products': products_list,
#             'total_products': total_products,
#             'total_pages': total_pages,
#             'current_page': page,
#             'per_page': PRODUCTS_PER_PAGE
#         }), 200

#     except Exception as e:
#         print(f"Error fetching products: {e}")
#         return jsonify({'error': 'Could not fetch product list from database.'}), 500

# # --- Search by URL ---
# @app.route('/api/search', methods=['POST'])
# def visual_search_url():
#     data = request.get_json()
#     is_valid, error = validate_data(data, ['image_url'])
#     if not is_valid:
#         return jsonify({'error': error}), 400

#     image_url = data['image_url'].strip()
#     query_vector = get_query_embedding(image_url, is_file=False)
#     if query_vector is None:
#         return jsonify({'error': 'Failed to process the provided image URL.'}), 400

#     try:
#         results = search_similar_products(query_vector, k=PRODUCTS_PER_PAGE)
#         return jsonify({'products': results}), 200
#     except Exception as e:
#         print(f"Visual search failed: {e}")
#         return jsonify({'error': 'An internal error occurred during the visual search.'}), 500

# # --- Search by File Upload ---
# @app.route('/api/upload_search', methods=['POST'])
# def visual_search_upload():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided (key: image).'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected image file.'}), 400
#     if not file.content_type.startswith('image/'):
#         return jsonify({'error': 'Invalid file type.'}), 400

#     query_vector = get_query_embedding(file.stream, is_file=True)
#     if query_vector is None:
#         return jsonify({'error': 'Failed to process uploaded image.'}), 400

#     try:
#         results = search_similar_products(query_vector, k=PRODUCTS_PER_PAGE)
#         return jsonify({'products': results}), 200
#     except Exception as e:
#         print(f"Visual search failed: {e}")
#         return jsonify({'error': 'An internal error occurred during the visual search.'}), 500

# # --- Flask Runner ---
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=False)





import os
from flask import Flask, request, jsonify
from pymongo import MongoClient
import bcrypt
from datetime import datetime
from bson.objectid import ObjectId
from flask_cors import CORS

# --- MobileNetV2 Model Imports ---
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
# ---------------------------------

# --- Search Imports ---
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import faiss
import random
# ----------------------

# --- Configuration and Setup ---

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- MongoDB Setup ---
# NOTE: Use os.environ.get("MONGO_URI") in production environment
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = 'fashion_matcher'
USERS_COLLECTION_NAME = 'users'
PRODUCTS_COLLECTION_NAME = 'visualproducts' # Updated to match your previous scripts

# --- Product and Search Configuration ---
PRODUCTS_PER_PAGE = 20
# 💡 IMPORTANT: MobileNetV2 output dimension
EMBEDDING_DIMENSION = 1280 

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 💡 IMPORTANT: Ensure these match the files you saved!
INDEX_FILENAME = os.path.join(BASE_DIR, "faiss_index.bin")
PRODUCT_IDS_FILE = os.path.join(BASE_DIR, "product_ids.npy") 

# Global components
users_collection = None
products_collection = None
# 💡 New global variables for MobileNetV2
mobilenet_model = None
mobilenet_transform = None
# -----------------------------------------
faiss_index = None
product_ids_map = None
search_device = "cpu"

# --- Database Connection ---
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')

    db = client[DB_NAME]
    users_collection = db[USERS_COLLECTION_NAME]
    products_collection = db[PRODUCTS_COLLECTION_NAME]
    users_collection.create_index("email", unique=True)

    print("✅ MongoDB: Connected to collections.")
    print("⚙️ MobileNetV2 and FAISS index will load lazily on first use.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to connect to MongoDB. Error: {e}")

# ----------------------------------------------------
## 💡 MOBILE-NET V2 SPECIFIC LOGIC
# ----------------------------------------------------

# --- Model Loading Helper ---
def load_mobilenetv2():
    """Loads MobileNetV2 model for feature extraction."""
    global mobilenet_model, mobilenet_transform
    
    # 1. Model Structure (Must match embedding script)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    
    # Remove classifier → get feature vector
    model.classifier = nn.Identity()

    full_model = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    ).to(search_device)
    
    # 2. Preprocessing Transforms (Must match embedding script)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    mobilenet_model = full_model
    mobilenet_transform = transform
    print(f"✅ MobileNetV2 (1280-d) loaded successfully on {search_device}.")

def ensure_mobilenet_loaded():
    """Lazily loads MobileNetV2 model and transform only once."""
    global mobilenet_model, mobilenet_transform
    if mobilenet_model is None:
        load_mobilenetv2()

# --- Lazy Loading FAISS ---
def ensure_faiss_index_loaded():
    """Lazily loads FAISS index and product ID map."""
    global faiss_index, product_ids_map
    if faiss_index is None or product_ids_map is None:
        print("⚙️ Loading FAISS index for the first time...")
        print(f"--- DEBUG: Attempting to load INDEX from: {INDEX_FILENAME}")
        print(f"--- DEBUG: Attempting to load IDs from: {PRODUCT_IDS_FILE}")
        if os.path.exists(INDEX_FILENAME) and os.path.exists(PRODUCT_IDS_FILE):
            # Load the FAISS index (assuming IndexFlatL2 or IndexFlatIP)
            faiss_index = faiss.read_index(INDEX_FILENAME) 
            # Load the product ID map (Numpy array of object IDs as strings)
            product_ids_map = np.load(PRODUCT_IDS_FILE) 
            print(f"✅ FAISS index loaded successfully with {faiss_index.ntotal} vectors.")
            print(f"--- DEBUG: Loaded {len(product_ids_map)} IDs into map.")
        else:
            print("⚠️ FAISS index or product ID map not found on server.")
            # It's better to raise an error that crashes the app if the core component is missing
            raise FileNotFoundError("FAISS index or ID map missing. Run the indexing script first!")

# --- Visual Search Logic ---
def get_query_embedding(image_source, is_file=False):
    """
    Fetches image (from URL or file), processes it with MobileNetV2, and returns L2-normalized embedding.
    """
    ensure_mobilenet_loaded()
    global mobilenet_model, mobilenet_transform, search_device
    
    try:
        # 1. Load Image
        if is_file:
            image = Image.open(image_source).convert("RGB")
        else:
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

        # 2. Preprocess and get tensor
        img_t = mobilenet_transform(image).unsqueeze(0).cpu()

        # 3. Generate Features
        with torch.no_grad():
            image_features = mobilenet_model(img_t).squeeze()
            
        # 4. Extract and L2-Normalize the vector
        query_vector = image_features.cpu().numpy().astype("float32")
        
        # L2-normalize the single query vector
        # norm_vector = query_vector / np.linalg.norm(query_vector)
        
        # Reshape to (1, D) for FAISS search
        return query_vector.reshape(1, -1)

    except Exception as e:
        print(f"[Error] Query image processing failed: {e}")
        return None

# In app.py
# In app.py
def search_similar_products(query_vector, k: int = 20):
    """Performs FAISS search and fetches metadata."""
    # Assume ensure_faiss_index_loaded() has been called successfully
    global faiss_index, product_ids_map, products_collection

    query_vector_faiss_ready = np.array(query_vector, dtype=np.float32, copy=True, order="C")
    
    # 1. Search FAISS Index
    D, I = faiss_index.search(query_vector_faiss_ready, k)
    faiss_indices = I[0]
    distances = D[0]

    product_id_strings = product_ids_map[faiss_indices].tolist()
    distances_list = distances.tolist()
    
    # 2. Prepare IDs for MongoDB (CRITICAL STEP)
    object_ids_to_query = []
    product_data_map = {}
    
    for pid_str in product_id_strings:
        try:
            # Convert the string from the .npy file into a MongoDB ObjectId
            object_ids_to_query.append(ObjectId(pid_str))
        except Exception as e:
            # Skip any malformed IDs in the FAISS map
            print(f"--- ERROR: Failed to convert ID '{pid_str}' to ObjectId: {e} ---")
            continue

    if not object_ids_to_query:
        print("--- DEBUG: Search returned 0 products (after ID conversion failed). ---")
        return []

    # 3. Retrieve all required product documents in a single bulk query
    # We query the native MongoDB _id field with the list of ObjectIds
    product_cursor = products_collection.find({"_id": {"$in": object_ids_to_query}})
    
    # Create a map from the string representation of _id to the document
    product_data_map = {str(doc.get("_id")): doc for doc in product_cursor}

    print(f"[Debug] Successfully retrieved {len(product_data_map)} products from MongoDB.")

    final_results = []
    all_distances = distances_list

    min_distance = min(all_distances)
    max_distance = max(all_distances)

    if max_distance == min_distance:
        distance_range = 1.0
    else:
        distance_range = max_distance - min_distance
    
    # 4. Combine FAISS results and MongoDB data
    for product_id_str, distance in zip(product_id_strings, all_distances):
        product = product_data_map.get(product_id_str)
        
        if product:
            # Score conversion: 1 / (1 + distance)
            normalized_distance = (distance - min_distance) / distance_range
            similarity_score = 1.0 - normalized_distance
        
        # Ensure the score is not slightly below 0 due to float errors
            similarity_score = max(0.0, similarity_score)
            
            # Note: Assuming price is missing and adding a placeholder or random value, 
            # if your database has a 'price' field, use product.get("price", 0.0)
            price = random.randint(50, 500) * 10 
            
            final_results.append({
                "id": product_id_str, 
                "name": product.get("productDisplayName", "Unknown Product"),
                "category": product.get("articleType", "N/A"),
                "price": price,
                "imageUrl": product.get("image_url"),
                "similarityScore": float(similarity_score),
            })

    print(f"--- DEBUG: Search returned {len(final_results)} products. ---")
    return final_results

# ----------------------------------------------------
## ENDPOINTS (Unchanged/Slightly Adjusted for MobileNetV2 logic)
# ----------------------------------------------------

# --- Utility Functions (Existing) ---
def validate_data(data, required_fields):
    if not data:
        return False, "Missing JSON data in request body."
    for field in required_fields:
        if field not in data or not str(data[field]).strip():
            return False, f"Missing or empty required field: {field}."
    return True, None

def generate_random_price():
    """Generates a random price between 500 and 5000 (increments of 10)."""
    return float(random.randint(50, 500) * 10)

# --- User Signup ---
@app.route('/api/signup', methods=['POST'])
def signup():
    if users_collection is None:
        return jsonify({'error': 'Database connection failed. Check server logs.'}), 503

    data = request.get_json()
    is_valid, error = validate_data(data, ['username', 'email', 'password'])
    # ... (rest of signup logic remains the same)
    if not is_valid:
        return jsonify({'error': error}), 400

    username = data['username'].strip()
    email = data['email'].strip().lower()
    password = data['password']

    if users_collection.find_one({'email': email}):
        return jsonify({'error': 'Email address already registered. Please try logging in.'}), 409

    try:
        password_bytes = password.encode('utf-8')
        hashed_password = bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode('utf-8')
    except Exception:
        return jsonify({'error': 'Internal server error during password processing.'}), 500

    new_user = {
        'username': username,
        'email': email,
        'password_hash': hashed_password,
        'created_at': datetime.utcnow()
    }

    try:
        users_collection.insert_one(new_user)
        return jsonify({'message': 'User created successfully.'}), 201
    except Exception:
        return jsonify({'error': 'Database error during account creation.'}), 500


# --- User Login ---
@app.route('/api/login', methods=['POST'])
def login():
    if users_collection is None:
        return jsonify({'error': 'Database connection failed. Check server logs.'}), 503

    data = request.get_json()
    is_valid, error = validate_data(data, ['email', 'password'])
    # ... (rest of login logic remains the same)
    if not is_valid:
        return jsonify({'error': error}), 400

    email = data['email'].strip().lower()
    password = data['password']
    user = users_collection.find_one({'email': email})

    if not user:
        return jsonify({'error': 'Invalid email or password.'}), 401

    try:
        stored_hash_bytes = user['password_hash'].encode('utf-8')
        password_bytes = password.encode('utf-8')

        if bcrypt.checkpw(password_bytes, stored_hash_bytes):
            user_id = str(user['_id'])
            simulated_token = f"jwt_user_{user_id}_{int(datetime.utcnow().timestamp())}"
            return jsonify({
                'message': 'Login successful.',
                'user_id': user_id,
                'username': user.get('username', 'User'),
                'token': simulated_token
            }), 200
        else:
            return jsonify({'error': 'Invalid email or password.'}), 401

    except Exception:
        return jsonify({'error': 'Internal server error during login verification.'}), 500

# --- Product Routes ---
@app.route('/api/products', methods=['GET'])
def get_products():
    if products_collection is None:
        return jsonify({'error': 'Product database connection failed.'}), 503

    try:
        page = int(request.args.get('page', 1))
        if page < 1:
            page = 1
    except ValueError:
        page = 1

    skip_count = (page - 1) * PRODUCTS_PER_PAGE
    try:
        # NOTE: Sorting by '_id' for stable pagination when no explicit 'id' field is present
        products_cursor = products_collection.find(
            {},
            {"_id": 1, "productDisplayName": 1, "articleType": 1, "price": 1, "image_url": 1}
        ).sort("_id", 1).skip(skip_count).limit(PRODUCTS_PER_PAGE)

        products_list = []
        for doc in products_cursor:
            price = generate_random_price()
            products_list.append({
                "id": str(doc.get("id", doc.get("_id"))), # Use the ObjectId string if no 'id' field exists
                "name": doc.get("productDisplayName", "N/A"),
                "category": doc.get("articleType", "N/A"),
                "price": price,
                "imageUrl": doc.get("image_url", "https://placehold.co/150x200/cccccc/000000?text=No+Image"),
                "similarityScore": 1.0
            })

        total_products = products_collection.count_documents({})
        total_pages = (total_products + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE

        return jsonify({
            'products': products_list,
            'total_products': total_products,
            'total_pages': total_pages,
            'current_page': page,
            'per_page': PRODUCTS_PER_PAGE
        }), 200

    except Exception as e:
        print(f"Error fetching products: {e}")
        return jsonify({'error': 'Could not fetch product list from database.'}), 500

# --- Search by URL ---
@app.route('/api/search', methods=['POST'])
def visual_search_url():
    data = request.get_json()
    is_valid, error = validate_data(data, ['image_url'])
    if not is_valid:
        return jsonify({'error': error}), 400

    image_url = data['image_url'].strip()
    print(f"--- DEBUG: Received URL: {image_url} ---")
    # query_vector is now (1, 1280) numpy array
    query_vector = get_query_embedding(image_url, is_file=False) 
    if query_vector is None:
        return jsonify({'error': 'Failed to process the provided image URL.'}), 400

    try:
        ensure_faiss_index_loaded()
        results = search_similar_products(query_vector, k=PRODUCTS_PER_PAGE)
        return jsonify({'products': results}), 200
    except FileNotFoundError:
        return jsonify({'error': 'Visual search index not found on server.'}), 500
    except Exception as e:
        print(f"Visual search failed: {e}")
        return jsonify({'error': 'An internal error occurred during the visual search.'}), 500

# --- Search by File Upload ---
@app.route('/api/upload_search', methods=['POST'])
def visual_search_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided (key: image).'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file.'}), 400
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'Invalid file type.'}), 400

    # query_vector is now (1, 1280) numpy array
    query_vector = get_query_embedding(file.stream, is_file=True) 
    if query_vector is None:
        return jsonify({'error': 'Failed to process uploaded image.'}), 400

    try:
        ensure_faiss_index_loaded()
        results = search_similar_products(query_vector, k=PRODUCTS_PER_PAGE)
        return jsonify({'products': results}), 200
    except FileNotFoundError:
        return jsonify({'error': 'Visual search index not found on server.'}), 500
    except Exception as e:
        print(f"Visual search failed: {e}")
        return jsonify({'error': 'An internal error occurred during the visual search.'}), 500

# --- Flask Runner ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
