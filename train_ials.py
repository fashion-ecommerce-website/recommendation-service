import os
import sys
import argparse
import logging
import pickle
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import psycopg2
import redis
from implicit.als import AlternatingLeastSquares
from dotenv import load_dotenv


# Load .env file nhưng không override environment variables đã có
# (ưu tiên environment variables từ docker-compose/system)
load_dotenv(override=False)


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------
# Default Params (override by CLI)
# ----------------------------
DEFAULT_ACTION_WEIGHTS = {
    "VIEW": 1.0,
    "LIKE": 3.0,
    "ADD_TO_CART": 5.0,
    "PURCHASE": 10.0,
}

DEFAULT_MODEL_PARAMS = {
    "factors": 64,
    "regularization": 0.01,
    "iterations": 20,
    "num_threads": 0,  # 0 = use all cores
}

# ----------------------------
# Data Loading
# ----------------------------

def get_db_conn(env: argparse.Namespace):
    """
    Create a psycopg2 connection using context manager in caller.
    """
    required = ["db_host", "db_port", "db_name", "db_user", "db_password"]
    for k in required:
        if getattr(env, k) in (None, ""):
            logger.error("Missing DB param: %s", k.upper())
            sys.exit(1)
    conn = psycopg2.connect(
        host=env.db_host,
        port=env.db_port,
        dbname=env.db_name,
        user=env.db_user,
        password=env.db_password,
        connect_timeout=10,
    )
    return conn


def read_interactions(env: argparse.Namespace) -> pd.DataFrame:
    """
    Read raw interactions from PostgreSQL into a DataFrame.
    Expecting a tall table with columns like:
      user_id BIGINT, product_id BIGINT, action_type TEXT, count INT (optional)
    If count is missing, default 1 per row.
    """
    sql = f"""
        SELECT user_id, product_id, action_type,
               CASE
                 WHEN count IS NULL THEN 1
                 ELSE count
               END AS count
        FROM interactions
    """
    with get_db_conn(env) as conn:
        df = pd.read_sql_query(sql, conn)
    if df.empty:
        logger.warning("No interactions found in table interactions.")
    else:
        logger.info("Read %d interaction rows from interactions.", len(df))
    return df


def aggregate_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Aggregate to (user_id, product_id, score) using action weights.
    """
    if df.empty:
        return pd.DataFrame(columns=["user_id", "product_id", "score"])

    # Normalize column names/types
    for col in ["user_id", "product_id", "action_type", "count"]:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # Clean action_type to lower
    df = df.copy()
    df["action_type"] = df["action_type"].str.lower().str.strip()

    # Map weight
    df["w"] = df["action_type"].map(weights).fillna(0.0)

    # Score = w * count
    df["score"] = df["w"] * df["count"].astype(float)

    # Aggregate per (user, product)
    agg = (
        df.groupby(["user_id", "product_id"], as_index=False)["score"]
        .sum()
        .query("score > 0")
    )

    logger.info(
        "Aggregated to %d (user, product) pairs; mean score=%.4f, max=%.4f",
        len(agg),
        agg["score"].mean() if not agg.empty else 0.0,
        agg["score"].max() if not agg.empty else 0.0,
    )
    return agg


# ----------------------------
# Sparse Matrix & Mappings
# ----------------------------

def build_mappings(agg: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build {user_id -> row_index}, {product_id -> col_index}.
    """
    user_ids = agg["user_id"].unique()
    product_ids = agg["product_id"].unique()

    user_ids_map = {uid: i for i, uid in enumerate(user_ids)}
    product_ids_map = {pid: j for j, pid in enumerate(product_ids)}

    logger.info(
        "Users: %d | Products: %d | Interactions: %d",
        len(user_ids_map),
        len(product_ids_map),
        len(agg),
    )
    return user_ids_map, product_ids_map


def build_user_item_matrix(
    agg: pd.DataFrame,
    user_ids_map: Dict[int, int],
    product_ids_map: Dict[int, int],
    alpha: float,
) -> coo_matrix:
    """
    Build user-item sparse matrix. Optionally apply confidence scaling:
      c = 1 + alpha * r_ui  (if alpha > 0)
    """
    if agg.empty:
        return coo_matrix((0, 0), dtype=np.float32)

    rows = agg["user_id"].map(user_ids_map).astype(int).values
    cols = agg["product_id"].map(product_ids_map).astype(int).values
    data = agg["score"].astype(float).values

    if alpha > 0:
        data = 1.0 + alpha * data

    n_users = len(user_ids_map)
    n_items = len(product_ids_map)
    mat = coo_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    nnz = mat.nnz
    density = nnz / float(max(1, n_users * n_items))
    logger.info("Built matrix shape=(%d, %d), nnz=%d, density=%.8f", n_users, n_items, nnz, density)
    return mat


# ----------------------------
# Train & Recommend
# ----------------------------

def train_ials(user_item: coo_matrix, model_params: Dict) -> AlternatingLeastSquares:
    """
    Train ALS on item-user matrix (transpose of user-item).
    """
    if user_item.shape[0] == 0 or user_item.shape[1] == 0:
        raise ValueError("Empty matrix. Cannot train model.")

    user_item_csr = user_item.tocsr()
    model = AlternatingLeastSquares(
        factors=model_params["factors"],
        regularization=model_params["regularization"],
        iterations=model_params["iterations"],
        num_threads=model_params.get("num_threads", 0),
    )
    logger.info(
        "Training IALS: factors=%d, reg=%.4f, iters=%d, threads=%d",
        model.factors, model.regularization, model.iterations, model.num_threads
    )
    # Train on user-item matrix as expected by implicit ALS
    model.fit(user_item_csr)
    logger.info("Training completed.")
    return model


def recommend_all_users(
    model: AlternatingLeastSquares,
    user_item_csr,
    top_n: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    For each user index, return (item_indices, scores).
    """
    n_users = user_item_csr.shape[0]
    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for u in range(n_users):
        # implicit.recommend requires user_items (csr row) for filtering if needed
        ids, scores = model.recommend(
            userid=u,
            user_items=user_item_csr[u],
            N=top_n,
            filter_already_liked_items=True,
            recalculate_user=True
        )
        results[u] = (ids, scores)
        if (u + 1) % 1000 == 0:
            logger.info("Recommended for %d/%d users...", u + 1, n_users)
    logger.info("Generated recommendations for %d users.", n_users)
    return results


# ----------------------------
# Persistence: Model & Redis
# ----------------------------

def save_model_and_meta(
    model: AlternatingLeastSquares,
    user_ids_map: Dict[int, int],
    product_ids_map: Dict[int, int],
    model_params: Dict,
    args: argparse.Namespace,
):
    os.makedirs(args.model_dir, exist_ok=True)
    out_path = os.path.join(args.model_dir, "ials_model_and_meta.pkl")
    payload = {
        "model": model,
        "user_ids_map": user_ids_map,
        "product_ids_map": product_ids_map,
        "model_params": model_params,
        "top_n": args.top_n,
        "alpha": args.alpha,
        "action_weights": args.action_weights,
        "table_name": "interactions",
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Saved model + meta to %s", out_path)


def write_most_popular_to_redis(
    env: argparse.Namespace,
    agg: pd.DataFrame,
):
    """
    Tính toán và lưu most popular items vào Redis với key rec:global:most-popular
    Score = tổng score của product_id (từ aggregated interactions)
    """
    if agg.empty:
        logger.warning("No aggregated data to calculate most popular items")
        return
    
    # Tính tổng score cho mỗi product_id
    product_scores = agg.groupby("product_id")["score"].sum().sort_values(ascending=False)
    
    pool = redis.ConnectionPool(
        host=env.redis_host,
        port=env.redis_port,
        db=env.redis_db,
        password=env.redis_password if env.redis_password else None,
        decode_responses=True,
        socket_timeout=10,
    )
    
    cache_key = f"{env.redis_prefix}:global:most-popular"
    
    with redis.Redis(connection_pool=pool) as r:
        # Clear previous list
        r.delete(cache_key)
        
        if len(product_scores) == 0:
            logger.warning("No product scores to write to Redis")
            return
        
        # ZADD expects mapping {member: score}
        # Score = tổng score của product (để sắp xếp theo popularity)
        mapping = {}
        for product_id, total_score in product_scores.items():
            mapping[str(int(product_id))] = float(total_score)
        
        r.zadd(cache_key, mapping)
        
        if env.redis_ttl > 0:
            r.expire(cache_key, env.redis_ttl)
        
        logger.info("Wrote %d most popular items to Redis key: %s", len(mapping), cache_key)
        logger.info("Top 10 most popular items: %s", list(product_scores.head(10).index.astype(int)))


def write_recs_to_redis(
    env: argparse.Namespace,
    recs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    user_ids_map: Dict[int, int],
    product_ids_map: Dict[int, int],
):
    """
    Write ZSET per user: key = rec:user:{real_user_id}
    Score in ZSET = rank (0..N-1). Client must ZRANGE to get ascending order.
    """
    # Build reverse maps
    inv_user = {idx: uid for uid, idx in user_ids_map.items()}
    inv_item = {idx: pid for pid, idx in product_ids_map.items()}

    pool = redis.ConnectionPool(
        host=env.redis_host,
        port=env.redis_port,
        db=env.redis_db,
        password=env.redis_password if env.redis_password else None,
        decode_responses=True,
        socket_timeout=10,
    )

    count_users = 0
    with redis.Redis(connection_pool=pool) as r:
        for u_idx, (item_indices, _scores) in recs.items():
            real_uid = inv_user[u_idx]
            key = f"{env.redis_prefix}:user:{real_uid}"

            # Clear previous list
            r.delete(key)

            if len(item_indices) == 0:
                # Keep empty key? Optionally skip.
                continue

            # ZADD expects mapping {member: score}
            mapping = {}
            for i_idx, score_val in zip(item_indices, _scores):
                real_pid = inv_item[int(i_idx)]
                # LƯU ĐIỂM SỐ THỰC TẾ
                mapping[str(real_pid)] = float(score_val)

            r.zadd(key, mapping)
            if env.redis_ttl > 0:
                r.expire(key, env.redis_ttl)

            count_users += 1
            if count_users % 1000 == 0:
                logger.info("Wrote Redis recs for %d users...", count_users)

    logger.info("Finished writing Redis for %d users.", count_users)


# ----------------------------
# CLI / Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train IALS and cache recommendations to Redis.")

    # DB
    p.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    p.add_argument("--db-port", default=int(os.getenv("DB_PORT", "5432")), type=int)
    p.add_argument("--db-name", default=os.getenv("DB_NAME"))
    p.add_argument("--db-user", default=os.getenv("DB_USER"))
    p.add_argument("--db-password", default=os.getenv("DB_PASSWORD"))

    # Redis
    p.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    p.add_argument("--redis-port", default=int(os.getenv("REDIS_PORT", "6379")), type=int)
    p.add_argument("--redis-db", default=int(os.getenv("REDIS_DB", "0")), type=int)
    p.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD"))
    p.add_argument("--redis-prefix", default=os.getenv("REDIS_PREFIX", "rec"))
    p.add_argument("--redis-ttl", default=int(os.getenv("REDIS_TTL", "604800")), type=int, help="TTL in seconds (default 7 days)")

    # Model
    p.add_argument("--factors", default=int(os.getenv("ALS_FACTORS", DEFAULT_MODEL_PARAMS["factors"])), type=int)
    p.add_argument("--regularization", default=float(os.getenv("ALS_REG", DEFAULT_MODEL_PARAMS["regularization"])), type=float)
    p.add_argument("--iterations", default=int(os.getenv("ALS_ITERS", DEFAULT_MODEL_PARAMS["iterations"])), type=int)
    p.add_argument("--num-threads", default=int(os.getenv("ALS_THREADS", "0")), type=int)

    # Data / Training
    p.add_argument("--alpha", default=float(os.getenv("IMPL_ALPHA", "10")), type=float, help="Confidence scaling: c = 1 + alpha * r_ui; 0 to disable.")
    p.add_argument("--top-n", default=int(os.getenv("TOP_N", "10")), type=int)
    p.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "model"))

    # Action weights (comma-separated "view=1,cart=5,purchase=10")
    p.add_argument("--action-weights", default=os.getenv("ACTION_WEIGHTS", "VIEW=1,ADD_TO_CART=5,LIKE=3,PURCHASE=8"))

    args = p.parse_args()

    # Parse action weights string
    weights = {}
    try:
        for token in str(args.action_weights).split(","):
            k, v = token.split("=")
            weights[k.strip().lower()] = float(v.strip())
    except Exception as e:
        logger.error("Failed to parse --action-weights: %s", e)
        sys.exit(1)
    args.action_weights = weights

    # Bundle model params
    args.model_params = {
        "factors": args.factors,
        "regularization": args.regularization,
        "iterations": args.iterations,
        "num_threads": args.num_threads,
    }
    return args


def main():
    args = parse_args()
    try:
        logger.info("Starting pipeline...")

        # 1) Load raw interactions
        df_raw = read_interactions(args)
        if df_raw.empty:
            logger.warning("No data to train. Exiting.")
            return

        # 2) Aggregate to scores
        agg = aggregate_scores(df_raw, args.action_weights)
        if agg.empty:
            logger.warning("No positive scores after aggregation. Exiting.")
            return

        # 3) Build mappings and sparse matrix (user-item)
        user_ids_map, product_ids_map = build_mappings(agg)
        user_item = build_user_item_matrix(agg, user_ids_map, product_ids_map, args.alpha).tocsr()

        # 4) Train ALS on item-user
        model = train_ials(user_item, args.model_params)

        # 5) Recommend per user
        recs = recommend_all_users(model, user_item, args.top_n)

        # 6) Save model + metadata
        save_model_and_meta(model, user_ids_map, product_ids_map, args.model_params, args)

        # 7) Write user recommendations to Redis
        write_recs_to_redis(args, recs, user_ids_map, product_ids_map)

        # 8) Write most popular items to Redis
        write_most_popular_to_redis(args, agg)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
