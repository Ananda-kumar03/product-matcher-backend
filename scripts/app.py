import os
from flask import Flask, request, jsonify
from pymongo import MongoClient
import bcrypt
from datetime import datetime
from bson.objectid import ObjectId
from flask_cors import CORS
# --- New Imports for Visual Search ---
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from sklearn.preprocessing import normalize
import faiss
import random  # For random price generation
# -------------------------------------

# --- Configuration and Setup ---

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# --- MongoDB Setup ---
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = 'fashion_matcher'
USERS_COLLECTION_NAME = 'users'
PRODUCTS_COLLECTION_NAME = 'products'

# --- Product and Search Configuration ---
PRODUCTS_PER_PAGE = 20
EMBEDDING_DIMENSION = 512

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILENAME = os.path.join(BASE_DIR, "faiss_index.bin")
PRODUCT_IDS_FILE = os.path.join(BASE_DIR, "product_ids.npy")
MODEL_NAME = "openai/clip-vit-base-patch32"

# Global components
users_collection = None
products_collection = None
search_model = None
search_processor = None
faiss_index = None
product_ids_map = None
search_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Database Connection (No model loading yet) ---
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')

    db = client[DB_NAME]
    users_collection = db[USERS_COLLECTION_NAME]
    products_collection = db[PRODUCTS_COLLECTION_NAME]
    users_collection.create_index("email", unique=True)

    print("✅ MongoDB: Connected to users and products collections. 'email' index ensured.")
    print("⚙️ Visual Search components will load lazily on first use (Render safe).")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to connect to MongoDB. Error: {e}")

# --- Utility Functions (Existing) ---
def validate_data(data, required_fields):
    if not data:
        return False, "Missing JSON data in request body."
    for field in required_fields:
        if field not in data or not str(data[field]).strip():
            return False, f"Missing or empty required field: {field}."
    return True, None

# --- Random Price Generator ---
def generate_random_price():
    """Generates a random price between 500 and 5000 (increments of 10)."""
    return float(random.randint(50, 500) * 10)

# --- Lazy Loading Helpers ---
def ensure_clip_model_loaded():
    """Lazily loads CLIP model and processor only once."""
    global search_model, search_processor, search_device
    if search_model is None or search_processor is None:
        print("⚙️ Loading CLIP model for the first time...")
        from transformers import CLIPProcessor, CLIPModel
        search_model = CLIPModel.from_pretrained(MODEL_NAME).to(search_device)
        search_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("✅ CLIP model loaded successfully.")

def ensure_faiss_index_loaded():
    """Lazily loads FAISS index and product ID map."""
    global faiss_index, product_ids_map
    if faiss_index is None or product_ids_map is None:
        print("⚙️ Loading FAISS index for the first time...")
        if os.path.exists(INDEX_FILENAME) and os.path.exists(PRODUCT_IDS_FILE):
            faiss_index = faiss.read_index(INDEX_FILENAME)
            product_ids_map = np.load(PRODUCT_IDS_FILE)
            print(f"✅ FAISS index loaded successfully with {faiss_index.ntotal} vectors.")
        else:
            print("⚠️ FAISS index or product ID map not found on server.")
            raise FileNotFoundError("Index or ID map missing.")

# --- Visual Search Logic ---
def get_query_embedding(image_source, is_file=False):
    """
    Fetches image (from URL or file), processes it, and returns normalized embedding.
    Lazily loads CLIP model + processor only once.
    """
    ensure_clip_model_loaded()

    try:
        if is_file:
            image = Image.open(image_source).convert("RGB")
        else:
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = search_processor(images=image, return_tensors="pt").to(search_device)

        with torch.no_grad():
            image_features = search_model.get_image_features(**inputs)

        query_vector = image_features.cpu().numpy().astype("float32")
        normalized_vector = normalize(query_vector, axis=1, copy=False)
        return normalized_vector

    except Exception as e:
        print(f"[Error] Query image processing failed: {e}")
        return None

def search_similar_products(query_vector, k: int = 20):
    """Performs FAISS search and fetches metadata. Lazily loads FAISS if needed."""
    ensure_faiss_index_loaded()
    global faiss_index, product_ids_map, products_collection

    query_vector_faiss_ready = np.array(query_vector, dtype=np.float32, copy=True, order="C").reshape(1, -1)
    if np.any(np.isnan(query_vector_faiss_ready)):
        return {"error": "Invalid query vector (NaN/Inf detected)."}

    D, I = faiss_index.search(query_vector_faiss_ready, k)
    faiss_indices = I[0]
    distances = D[0]

    result_product_ids_int = product_ids_map[faiss_indices].tolist()
    distances_list = distances.tolist()

    product_cursor = products_collection.find({"id": {"$in": result_product_ids_int}})
    product_data_map = {doc.get("id") or doc.get("_id"): doc for doc in product_cursor}

    if not product_data_map and result_product_ids_int:
        result_product_ids_str = [str(pid) for pid in result_product_ids_int]
        product_cursor_str = products_collection.find({"id": {"$in": result_product_ids_str}})
        product_data_map = {doc.get("id") or doc.get("_id"): doc for doc in product_cursor_str}

    final_results = []
    for product_id_int, distance in zip(result_product_ids_int, distances_list):
        product = product_data_map.get(product_id_int) or product_data_map.get(str(product_id_int))
        if product:
            similarity_score = 1.0 / (1.0 + distance)
            price = generate_random_price()
            final_results.append({
                "id": str(product.get("id", product.get("_id"))),
                "name": product.get("productDisplayName", "Unknown Product"),
                "category": product.get("articleType", "N/A"),
                "price": price,
                "imageUrl": product.get("image_url"),
                "similarityScore": float(similarity_score),
            })

    final_results.sort(key=lambda x: x["similarityScore"], reverse=True)
    return final_results

# --- User Signup ---
@app.route('/api/signup', methods=['POST'])
def signup():
    if users_collection is None:
        return jsonify({'error': 'Database connection failed. Check server logs.'}), 503

    data = request.get_json()
    is_valid, error = validate_data(data, ['username', 'email', 'password'])
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
        products_cursor = products_collection.find(
            {},
            {"id": 1, "productDisplayName": 1, "articleType": 1, "price": 1, "image_url": 1}
        ).sort("id", 1).skip(skip_count).limit(PRODUCTS_PER_PAGE)

        products_list = []
        for doc in products_cursor:
            price = generate_random_price()
            products_list.append({
                "id": str(doc.get("id", doc.get("_id"))),
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
    query_vector = get_query_embedding(image_url, is_file=False)
    if query_vector is None:
        return jsonify({'error': 'Failed to process the provided image URL.'}), 400

    try:
        results = search_similar_products(query_vector, k=PRODUCTS_PER_PAGE)
        return jsonify({'products': results}), 200
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

    query_vector = get_query_embedding(file.stream, is_file=True)
    if query_vector is None:
        return jsonify({'error': 'Failed to process uploaded image.'}), 400

    try:
        results = search_similar_products(query_vector, k=PRODUCTS_PER_PAGE)
        return jsonify({'products': results}), 200
    except Exception as e:
        print(f"Visual search failed: {e}")
        return jsonify({'error': 'An internal error occurred during the visual search.'}), 500

# --- Flask Runner ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
