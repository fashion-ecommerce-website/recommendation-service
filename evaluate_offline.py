import os
import sys
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Set
import redis
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix

# -------------------
# Load env (re-use logic)
# -------------------
load_dotenv()

# -------------
# Logging Setup
# -------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------
# CLI Parameters
# ---------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model offline (IALS, MostPopular, Random v.v.)")
    
    # DB
    p.add_argument("--db-host", default=os.getenv("DB_HOST"))
    p.add_argument("--db-port", default=os.getenv("DB_PORT", "5432"), type=int)
    p.add_argument("--db-name", default=os.getenv("DB_NAME"))
    p.add_argument("--db-user", default=os.getenv("DB_USER"))
    p.add_argument("--db-password", default=os.getenv("DB_PASSWORD"))
    
    # Redis
    p.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    p.add_argument("--redis-port", default=os.getenv("REDIS_PORT", "6379"), type=int)
    p.add_argument("--redis-db", default=os.getenv("REDIS_DB", "0"), type=int)
    p.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD"))
    p.add_argument("--redis-prefix", default=os.getenv("REDIS_PREFIX", "rec:user:"))
    
    # Model
    p.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "model"))
    p.add_argument("--model-file", default="ials_model_and_meta.pkl")

    # Eval Specific
    p.add_argument("--cutoff", required=True, help="Datetime cutoff train/test theo định dạng YYYY-MM-DD")
    p.add_argument("--k", type=int, default=10, help="Top-K recommendation")
    p.add_argument("--topk-list", default=None, help="Danh sách các K phân cách bởi dấu phẩy, vd: '5,10,20'")
    p.add_argument("--gt-events", default="purchase", help="Các event tính GT, phân cách bởi dấu phẩy")
    p.add_argument("--include-add-to-cart", action="store_true", help="Bao gồm event 'add_to_cart' vào groundtruth test nếu bật")
    p.add_argument("--use-redis", type=lambda x: x.lower()=='true', default=False, help="True: lấy rec từ Redis; False: infer từ model")
    p.add_argument("--output-dir", default="./eval_out")
    p.add_argument("--seed", type=int, default=42)

    # Có thể bổ sung thêm các tham số khác nếu muốn
    return p.parse_args()

def get_db_conn(args):
    import psycopg2
    required = ["db_host", "db_port", "db_name", "db_user", "db_password"]
    for k in required:
        if getattr(args, k) in (None, ""):
            logger.error(f"Missing DB param: {k.upper()}")
            sys.exit(1)
    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        connect_timeout=10,
    )
    return conn

def read_interactions(args) -> pd.DataFrame:
    # Bạn có thể override hoặc bổ sung SQL ở đây tùy case thực tế
    sql = f"""
        SELECT user_id, product_id, action_type, 
               CASE WHEN count IS NULL THEN 1 ELSE count END AS count, 
               created_at
        FROM interactions
    """
    with get_db_conn(args) as conn:
        df = pd.read_sql_query(sql, conn)
    if df.empty:
        logger.warning("No interactions found in table.")
    else:
        logger.info(f"Read {len(df)} interaction rows from DB.")
    return df

def load_model_and_meta(model_dir, model_file):
    path = os.path.join(model_dir, model_file)
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    model = payload['model']
    user_ids_map = payload['user_ids_map']
    product_ids_map = payload['product_ids_map']
    # Có thể lấy thêm meta nếu cần
    return model, user_ids_map, product_ids_map, payload

def split_train_test(df: pd.DataFrame, cutoff: str, gt_events: List[str], include_add_to_cart: bool):
    # Coi cột thời gian tên là created_at, chỉnh lại nếu tên khác
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')
    train = df[df['created_at'] <= cutoff]
    test = df[df['created_at'] > cutoff]
    
    if include_add_to_cart:
        gt_events = list(set(gt_events + ['add_to_cart']))
    gt_events = set([e.strip().lower() for e in gt_events])
    
    # Các item user đã thấy ở Train (mọi event)
    train_seen = train.groupby('user_id')['product_id'].apply(set).to_dict()
    # Các item user tương tác ở Test, chỉ lấy action thuộc gt_events
    test = test[test['action_type'].str.lower().isin(gt_events)]
    test_positives = test.groupby('user_id')['product_id'].apply(set).to_dict()
    return train, test, train_seen, test_positives

# -----------------
# Build Train Matrix
# -----------------
def build_train_user_item_matrix(train_df, user_ids_map, product_ids_map):
    """
    Build user-item sparse matrix from train data using model's mappings.
    Only include users/products that exist in the mappings.
    """
    if train_df.empty:
        n_users = len(user_ids_map)
        n_items = len(product_ids_map)
        return csr_matrix((n_users, n_items), dtype=np.float32)
    
    # Filter to only users/products in mappings
    train_filtered = train_df[
        train_df['user_id'].isin(user_ids_map.keys()) & 
        train_df['product_id'].isin(product_ids_map.keys())
    ].copy()
    
    if train_filtered.empty:
        n_users = len(user_ids_map)
        n_items = len(product_ids_map)
        return csr_matrix((n_users, n_items), dtype=np.float32)
    
    # Map to indices
    rows = train_filtered['user_id'].map(user_ids_map).astype(int).values
    cols = train_filtered['product_id'].map(product_ids_map).astype(int).values
    
    # Use count as score (or score if exists)
    if 'score' in train_filtered.columns:
        data = train_filtered['score'].astype(float).values
    else:
        data = train_filtered['count'].astype(float).values
    
    n_users = len(user_ids_map)
    n_items = len(product_ids_map)
    mat = coo_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    return mat.tocsr()

# -----------------
# Redis Rec Loader
# -----------------
def get_user_recs_from_redis(user_id, k, redis_client, redis_prefix, exclude_items=None):
    key = f"{redis_prefix}{user_id}"
    items = redis_client.zrange(key, 0, k * 3 - 1) # Lấy số lượng nhiều hơn để dự phòng lọc
    if exclude_items:
        recs = [int(i) for i in items if int(i) not in exclude_items][:k]
    else:
        recs = [int(i) for i in items][:k]
    return recs

# ------------------
# Model Rec Inference
# ------------------
def get_user_recs_from_model(user_id, k, model, user_ids_map, product_ids_map, train_user_item_csr, exclude_items=None):
    """
    Get recommendations from model for a user.
    train_user_item_csr: CSR matrix of user-item interactions from training data.
    """
    if user_id not in user_ids_map:
        return []
    user_idx = user_ids_map[user_id]
    
    # Get user's row from train matrix (may be empty if user not in train)
    if train_user_item_csr is not None and user_idx < train_user_item_csr.shape[0]:
        user_items = train_user_item_csr[user_idx]
    else:
        # Create empty sparse matrix with correct shape
        n_items = len(product_ids_map)
        user_items = csr_matrix((1, n_items), dtype=np.float32)
    
    ids, scores = model.recommend(
        userid=user_idx,
        user_items=user_items,
        N=k * 3,
        filter_already_liked_items=False,  # Tự lọc thủ công (dùng exclude_items)
        recalculate_user=True
    )
    inv_product_map = {v: k for k, v in product_ids_map.items()}
    items = [inv_product_map[idx] for idx in ids]
    if exclude_items:
        items = [i for i in items if i not in exclude_items]
    return items[:k]

# ---------------------
# MostPopular Generator
# ---------------------
def generate_most_popular(train_df, k, exclude_items=None):
    freq = train_df.groupby("product_id")['score'].sum().sort_values(ascending=False)
    items = list(freq.index)
    if exclude_items:
        items = [i for i in items if i not in exclude_items]
    return items[:k]

# -----------------
# Random Generator
# -----------------
def generate_random(item_pool, k, exclude_items=None):
    sample = list(item_pool)
    if exclude_items:
        sample = [i for i in sample if i not in exclude_items]
    if len(sample) <= k:
        return sample
    return list(np.random.choice(sample, size=k, replace=False))

def hitrate_at_k(recs, positives):
    return int(len(set(recs) & set(positives)) > 0)

def precision_at_k(recs, positives):
    hits = len(set(recs) & set(positives))
    return hits / len(recs) if recs else 0.0

def recall_at_k(recs, positives):
    hits = len(set(recs) & set(positives))
    return hits / len(positives) if positives else 0.0

def ndcg_at_k(recs, positives):
    dcg = 0.0
    for i, item in enumerate(recs):
        if item in positives:
            dcg += 1 / np.log2(i + 2)
    # ideal dcg
    ideal = min(len(positives), len(recs))
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def coverage_at_k(all_recs, all_items):
    unique_rec_items = set([item for user_recs in all_recs for item in user_recs])
    return len(unique_rec_items) / len(all_items) if all_items else 0.0

# Lưu ý tên model_key: rec_redis, rec_model, rec_mostpop, rec_random gắn với từng user_results
ALL_MODELS = ['rec_redis', 'rec_model', 'rec_mostpop', 'rec_random']


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model/meta/mapping
    model, user_ids_map, product_ids_map, meta = load_model_and_meta(args.model_dir, args.model_file)
    
    # Đọc interactions từ DB
    df = read_interactions(args)
    if df.empty:
        logger.error("Không có dữ liệu interactions, dừng evaluation.")
        sys.exit(1)

    # Chia tập Train/Test
    gt_events = [e.strip() for e in args.gt_events.split(",") if e.strip()]
    train, test, train_seen, test_positives = split_train_test(df, args.cutoff, gt_events, args.include_add_to_cart)

    logger.info(f"Train users: {len(train['user_id'].unique())} | Test users: {len(test['user_id'].unique())}")
    logger.info(f"#Users trong test positives: {len(test_positives)} (filtered by gt-events)")

    # Chuẩn bị Redis nếu cần
    redis_client = None
    if args.use_redis:
        pool = redis.ConnectionPool(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            password=args.redis_password if args.redis_password else None,
            decode_responses=True,
            socket_timeout=10,
        )
        redis_client = redis.Redis(connection_pool=pool)
    
    all_items = set(train['product_id'].unique())
    train_scored = train.copy()
    if 'score' not in train_scored.columns:
        train_scored['score'] = train_scored['count']
    
    # Build train user-item matrix for model recommendations
    train_user_item_csr = None
    if not args.use_redis:
        logger.info("Building train user-item matrix for model recommendations...")
        train_user_item_csr = build_train_user_item_matrix(train_scored, user_ids_map, product_ids_map)
        logger.info(f"Train matrix shape: {train_user_item_csr.shape}, nnz: {train_user_item_csr.nnz}")
    
    users_eval = list(test_positives.keys())
    k_eval = args.k
    user_results = []
    missing_gt, missing_key_redis = 0, 0
    for user_id in tqdm(users_eval, desc="Eval reco by user"):
        exclude = train_seen.get(user_id, set())
        try:
            positives = set(test_positives[user_id])
        except KeyError:
            missing_gt += 1
            continue
        recs = {}
        for model_key in ALL_MODELS:
            data = []
            if model_key == 'rec_redis' and args.use_redis and redis_client:
                try:
                    data = get_user_recs_from_redis(user_id, k_eval, redis_client, args.redis_prefix, exclude_items=exclude)
                    if not data: missing_key_redis += 1
                except Exception:
                    missing_key_redis += 1
                    data = []
            elif model_key == 'rec_model' and not args.use_redis:
                try:
                    data = get_user_recs_from_model(user_id, k_eval, model, user_ids_map, product_ids_map, train_user_item_csr, exclude_items=exclude)
                except Exception:
                    data = []
            elif model_key == 'rec_mostpop':
                data = generate_most_popular(train_scored, k_eval, exclude_items=exclude)
            elif model_key == 'rec_random':
                data = generate_random(all_items, k_eval, exclude_items=exclude)
            recs[model_key] = data
        user_results.append({
            'user_id': user_id,
            **recs,
            'test_positives': list(positives),
            'train_seen': list(exclude),
        })
    # ========== Tính metrics per-user & macro ============
    per_user_metric_rows = []
    model_metrics = {m: {'hitrate': [], 'precision': [], 'recall': [], 'ndcg': []} for m in ALL_MODELS}
    coverages = {m: [] for m in ALL_MODELS}
    filtered_users = 0
    for row in user_results:
        positives = set(row['test_positives'])
        if len(positives) == 0:
            filtered_users += 1
            continue
        for model_key in ALL_MODELS:
            recs = row[model_key]
            hit = hitrate_at_k(recs, positives)
            prec = precision_at_k(recs, positives)
            rec = recall_at_k(recs, positives)
            ndcg = ndcg_at_k(recs, positives)
            per_user_metric_rows.append({
                "user_id": row['user_id'],
                "model": model_key,
                "hits": hit,
                "precision@k": prec,
                "recall@k": rec,
                "ndcg@k": ndcg,
                "reco_size": len(recs),
                "positives": len(positives)
            })
            model_metrics[model_key]['hitrate'].append(hit)
            model_metrics[model_key]['precision'].append(prec)
            model_metrics[model_key]['recall'].append(rec)
            model_metrics[model_key]['ndcg'].append(ndcg)
            coverages[model_key].append(recs)
    leaderboard_rows = []
    n_users_eval = len(user_results) - filtered_users
    for model_key in ALL_MODELS:
        leaderboard_rows.append({
            'model': model_key,
            'k': k_eval,
            'users_eval': n_users_eval,
            'hitrate@k': np.mean(model_metrics[model_key]['hitrate']) if model_metrics[model_key]['hitrate'] else 0.0,
            'precision@k': np.mean(model_metrics[model_key]['precision']) if model_metrics[model_key]['precision'] else 0.0,
            'recall@k': np.mean(model_metrics[model_key]['recall']) if model_metrics[model_key]['recall'] else 0.0,
            'ndcg@k': np.mean(model_metrics[model_key]['ndcg']) if model_metrics[model_key]['ndcg'] else 0.0,
            'coverage@k': coverage_at_k(coverages[model_key], all_items) if coverages[model_key] else 0.0
        })
    # ========== Xuất file vào folder theo ngày, tên file có giờ ==============
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")  # YYYYMMDD cho folder
    time_str = now.strftime("%H%M%S")   # HHMMSS cho tên file
    
    eval_output_dir = os.path.join(args.output_dir, date_str)
    os.makedirs(eval_output_dir, exist_ok=True)
    
    leaderboard_path = os.path.join(eval_output_dir, f"leaderboard_{time_str}.csv")
    per_user_metrics_path = os.path.join(eval_output_dir, f"per_user_metrics_{time_str}.csv")
    summary_path = os.path.join(eval_output_dir, f"summary_{time_str}.json")

    pd.DataFrame(leaderboard_rows).to_csv(leaderboard_path, index=False)
    pd.DataFrame(per_user_metric_rows).to_csv(per_user_metrics_path, index=False)

    # summary.json chứa các tham số, #user, #item, seed, lift...
    summary = {
        "cutoff": args.cutoff,
        "k": args.k,
        "seed": args.seed,
        "users_eval": n_users_eval,
        "users_filtered": filtered_users,
        "users_missing_gt": missing_gt,
        "users_missing_key_redis": missing_key_redis,
        "#users_train": len(train['user_id'].unique()),
        "#users_test": len(test['user_id'].unique()),
        "#items": len(all_items),
        "lift_vs_mostpopular": {
            "ndcg@k": leaderboard_rows[0]['ndcg@k'] - leaderboard_rows[2]['ndcg@k'],
            "recall@k": leaderboard_rows[0]['recall@k'] - leaderboard_rows[2]['recall@k'],
        } if leaderboard_rows and len(leaderboard_rows) >= 3 else {},
        "args": vars(args)
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\nEvaluation report saved to folder: {eval_output_dir}")
    logger.info(f"Files: leaderboard_{time_str}.csv, per_user_metrics_{time_str}.csv, summary_{time_str}.json")
    logger.info(f"Số user bị loại do không có positives: {filtered_users}, không có key Redis: {missing_key_redis}")

if __name__ == "__main__":
    main()
