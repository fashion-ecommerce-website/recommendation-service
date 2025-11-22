"""
Python AI-Service - "Cỗ máy tính toán" (Dumb Service)
Chỉ cung cấp các công cụ tính toán, không có logic if-else
"""
import os
import sys
import logging
import pickle
import subprocess
import threading
from typing import List, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
import redis
import numpy as np
from implicit.als import AlternatingLeastSquares
from dotenv import load_dotenv

# Load .env file nhưng không override environment variables đã có
# (ưu tiên environment variables từ docker-compose/system)
load_dotenv(override=False)

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Redis connection
redis_client = None
model_data = None
model = None
product_ids_map = None
user_ids_map = None

# Training lock
training_lock = threading.Lock()
is_training = False


def init_redis():
    """Khởi tạo Redis connection"""
    global redis_client
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=10,
        )
        redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        sys.exit(1)


def load_model():
    """Load model từ file pickle"""
    global model_data, model, product_ids_map, user_ids_map
    
    # Sử dụng relative path từ thư mục script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.getenv("MODEL_DIR", "model")
    # Nếu model_dir là relative path, tạo absolute path từ script_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(script_dir, model_dir)
    model_path = os.path.join(model_dir, "ials_model_and_meta.pkl")
    
    if not os.path.exists(model_path):
        logger.error("Model file not found: %s", model_path)
        logger.error("Please train model first: python train_ials.py")
        sys.exit(1)
    
    logger.info("Loading model from %s...", model_path)
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e) or "numpy" in str(e).lower():
            logger.error("Model compatibility error: Model was trained with different NumPy version")
            logger.error("Current NumPy version: %s", np.__version__)
            logger.error("Please retrain model with current NumPy version: python train_ials.py")
            sys.exit(1)
        raise
    
    model = model_data["model"]
    product_ids_map = model_data["product_ids_map"]
    user_ids_map = model_data["user_ids_map"]
    
    logger.info("Model loaded: factors=%d, regularization=%.4f",
                model.factors, model.regularization)
    logger.info("Users: %d, Products: %d",
                len(user_ids_map), len(product_ids_map))


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "ai-recommendation"})


def run_training_script():
    """Chạy script training trong background thread"""
    global is_training
    
    try:
        # Lấy đường dẫn script training
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, "train_ials.py")
        
        if not os.path.exists(train_script):
            logger.error("Training script not found: %s", train_script)
            return
        
        # Xác định Python executable để sử dụng
        # Ưu tiên sử dụng Python từ venv nếu có, nếu không thì dùng sys.executable
        python_exe = sys.executable
        # Kiểm tra venv Python (Windows và Linux/Mac)
        if sys.platform == "win32":
            venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
        else:
            venv_python = os.path.join(script_dir, "venv", "bin", "python")
        
        if os.path.exists(venv_python):
            python_exe = venv_python
            logger.info("Using venv Python: %s", python_exe)
        else:
            logger.info("Using system Python: %s", python_exe)
        
        logger.info("Starting training script: %s", train_script)
        
        # Chạy script với subprocess, sử dụng cùng Python environment
        # Đảm bảo sử dụng cùng environment variables
        env = os.environ.copy()
        process = subprocess.Popen(
            [python_exe, train_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=script_dir,
            env=env
        )
        
        # Log output từ script training
        for line in process.stdout:
            logger.info("[TRAINING] %s", line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            logger.info("Training completed successfully")
            # Reload model sau khi training xong
            try:
                load_model()
                logger.info("Model reloaded successfully after training")
            except Exception as e:
                logger.error("Failed to reload model after training: %s", e)
        else:
            logger.error("Training failed with return code: %d", process.returncode)
            
    except Exception as e:
        logger.exception("Error running training script: %s", e)
    finally:
        with training_lock:
            is_training = False
            logger.info("Training lock released")


@app.route("/admin/retrain-model", methods=["POST"])
def retrain_model():
    """
    Endpoint để admin/cron job gọi để retrain model
    Trigger script train_ials.py trong background thread
    Trả về OK ngay lập tức, không đợi training hoàn thành
    """
    global is_training
    
    # Kiểm tra xem có đang training không
    with training_lock:
        if is_training:
            logger.warning("Training already in progress, ignoring request")
            return jsonify({
                "status": "busy",
                "message": "Training already in progress"
            }), 409
        
        is_training = True
    
    logger.info("Retrain model request received, starting training in background")
    
    # Chạy training trong background thread
    training_thread = threading.Thread(target=run_training_script, daemon=True)
    training_thread.start()
    
    return jsonify({
        "status": "ok",
        "message": "Training started in background"
    })


@app.route("/recs/user/<int:user_id>", methods=["GET"])
def get_user_recommendations(user_id: int):
    """
    API Gợi ý Cá nhân hóa
    Chỉ làm 1 việc: Lấy gợi ý đã tính trước cho user_id từ Redis
    Nếu user_id không có trong Redis, trả về 404 Not Found
    """
    redis_prefix = os.getenv("REDIS_PREFIX", "rec")
    cache_key = f"{redis_prefix}:user:{user_id}"
    
    try:
        # Lấy từ Redis (sorted set, score = rank)
        product_ids = redis_client.zrange(cache_key, 0, -1)
        
        if not product_ids:
            logger.warning("No recommendations found for user_id: %d", user_id)
            return jsonify({"error": "No recommendations found"}), 404
        
        # Convert to integers
        result = [int(pid) for pid in product_ids]
        logger.info("Found %d recommendations for user_id: %d", len(result), user_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error getting recommendations for user_id %d: %s", user_id, e)
        return jsonify({"error": str(e)}), 500


@app.route("/recs/item/<int:item_id>", methods=["GET"])
def get_similar_items(item_id: int):
    """
    API Gợi ý Tương tự
    Chỉ làm 1 việc: Gọi "nóng" model.similar_items(item_id) và trả về 10 sản phẩm tương tự
    """
    limit = request.args.get("limit", 10, type=int)
    
    try:
        # Kiểm tra item_id có trong model không
        if item_id not in product_ids_map:
            logger.warning("Item_id %d not found in model", item_id)
            return jsonify({"error": "Item not found in model"}), 404
        
        item_idx = product_ids_map[item_id]
        
        # Lấy similar items từ model
        # implicit.als có method similar_items() để tính similarity
        similar_items, scores = model.similar_items(item_idx, N=limit)
        
        # Convert indices back to product IDs
        inv_product_map = {idx: pid for pid, idx in product_ids_map.items()}
        result = [int(inv_product_map[int(idx)]) for idx in similar_items]
        
        logger.info("Found %d similar items for item_id: %d", len(result), item_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error getting similar items for item_id %d: %s", item_id, e)
        return jsonify({"error": str(e)}), 500


@app.route("/recs/global/most-popular", methods=["GET"])
def get_most_popular():
    """
    API Gợi ý Phổ biến
    Trả về 10 món "hot" nhất (được lưu trong Redis key rec:global:most-popular)
    Đây là fallback cuối cùng cho user "lạnh"
    """
    limit = request.args.get("limit", 10, type=int)
    redis_prefix = os.getenv("REDIS_PREFIX", "rec")
    cache_key = f"{redis_prefix}:global:most-popular"
    
    try:
        # Lấy từ Redis (ZSET, sorted by score descending)
        # ZRANGE với desc=True để lấy items có score cao nhất
        product_ids = redis_client.zrange(cache_key, 0, limit - 1, desc=True, withscores=False)
        
        if not product_ids:
            logger.warning("No popular items found in cache key: %s", cache_key)
            # Fallback: trả về empty list
            return jsonify([])
        
        # Convert to integers
        result = [int(pid) for pid in product_ids]
        logger.info("Found %d popular items from key: %s", len(result), cache_key)
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error getting popular items from key %s: %s", cache_key, e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Khởi tạo
    init_redis()
    load_model()
    
    # Chạy server
    port = int(os.getenv("API_PORT", "5000"))
    logger.info("Starting AI Recommendation API service on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)

