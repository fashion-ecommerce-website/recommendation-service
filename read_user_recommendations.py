"""
Script để đọc recommendations từ Redis cho một user và hiển thị thông tin sản phẩm
"""
import os
import sys
import argparse
import logging
import redis
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_redis_connection(redis_host, redis_port, redis_db, redis_password, redis_prefix):
    """Kết nối đến Redis"""
    pool = redis.ConnectionPool(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password if redis_password else None,
        decode_responses=True,
        socket_timeout=10,
    )
    return redis.Redis(connection_pool=pool), redis_prefix


def get_db_connection(db_host, db_port, db_name, db_user, db_password):
    """Kết nối đến PostgreSQL"""
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        connect_timeout=10,
    )
    return conn


def get_recommendations_from_redis(redis_client, redis_prefix, user_id, limit=None):
    """
    Đọc recommendations từ Redis cho một user
    
    Args:
        redis_client: Redis client
        redis_prefix: Prefix cho Redis key
        user_id: ID của user
        limit: Số lượng recommendations cần lấy (None = lấy tất cả)
    
    Returns:
        List of tuples: [(product_id, score), ...] được sắp xếp theo score (giảm dần - tốt nhất trước)
    """
    key = f"{redis_prefix}:user:{user_id}"
    
    # Kiểm tra key có tồn tại không
    if not redis_client.exists(key):
        logger.warning("Key '%s' không tồn tại trong Redis", key)
        return []
    
    # Lấy tất cả members với scores từ ZSET (sắp xếp theo score GIẢM DẦN - score cao = tốt hơn)
    # ZREVRANGE key 0 -1 WITHSCORES (reverse range - từ cao xuống thấp)
    items = redis_client.zrevrange(key, 0, -1, withscores=True)
    
    if limit:
        items = items[:limit]
    
    # Convert sang list of tuples (product_id, score)
    recommendations = [(int(product_id), float(score)) for product_id, score in items]
    
    logger.info("Đọc được %d recommendations từ Redis cho user %d", len(recommendations), user_id)
    return recommendations


def get_products_info(db_conn, product_ids):
    """
    Lấy thông tin sản phẩm từ database
    
    Args:
        db_conn: PostgreSQL connection
        product_ids: List of product IDs
    
    Returns:
        Dict: {product_id: {'id': id, 'title': title}, ...}
    """
    if not product_ids:
        return {}
    
    # Query để lấy id và title của các sản phẩm
    query = """
        SELECT id, title
        FROM products
        WHERE id = ANY(%s)
        AND is_active = TRUE
        ORDER BY id
    """
    
    with db_conn.cursor() as cur:
        cur.execute(query, (product_ids,))
        rows = cur.fetchall()
    
    # Tạo dictionary: product_id -> {id, title}
    products_info = {row[0]: {'id': row[0], 'title': row[1]} for row in rows}
    
    logger.info("Lấy được thông tin của %d/%d sản phẩm từ database", len(products_info), len(product_ids))
    return products_info


def display_recommendations(user_id, recommendations, products_info):
    """
    Hiển thị recommendations ra console
    
    Args:
        user_id: ID của user
        recommendations: List of tuples (product_id, score)
        products_info: Dict {product_id: {'id': id, 'title': title}, ...}
    """
    print("\n" + "=" * 80)
    print(f"RECOMMENDATIONS CHO USER ID: {user_id}")
    print("=" * 80)
    
    if not recommendations:
        print("❌ Không có recommendations nào cho user này.")
        return
    
    print(f"\nTổng số recommendations: {len(recommendations)}\n")
    print(f"{'STT':<5} {'Product ID':<12} {'Score':<15} {'Tên sản phẩm':<50}")
    print("-" * 85)
    
    missing_products = []
    
    for idx, (product_id, score) in enumerate(recommendations, 1):
        if product_id in products_info:
            product = products_info[product_id]
            title = product['title']
            # Giới hạn độ dài title để dễ đọc
            if len(title) > 47:
                title = title[:44] + "..."
            print(f"{idx:<5} {product_id:<12} {score:<15.6f} {title:<50}")
        else:
            print(f"{idx:<5} {product_id:<12} {score:<15.6f} {'❌ Không tìm thấy trong DB':<50}")
            missing_products.append(product_id)
    
    print("-" * 85)
    
    if missing_products:
        logger.warning("Có %d sản phẩm không tìm thấy trong database: %s", 
                      len(missing_products), missing_products[:10])
    
    print("\n" + "=" * 80 + "\n")


def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser(
        description="Đọc recommendations từ Redis cho một user và hiển thị thông tin sản phẩm"
    )
    
    # User ID (required)
    p.add_argument("--user-id", type=int, required=True, help="ID của user cần xem recommendations")
    
    # Limit
    p.add_argument("--limit", type=int, default=None, help="Số lượng recommendations cần lấy (mặc định: tất cả)")
    
    # Redis config
    p.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    p.add_argument("--redis-port", default=int(os.getenv("REDIS_PORT", "6379")), type=int)
    p.add_argument("--redis-db", default=int(os.getenv("REDIS_DB", "0")), type=int)
    p.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD"))
    p.add_argument("--redis-prefix", default=os.getenv("REDIS_PREFIX", "rec"))
    
    # DB config
    p.add_argument("--db-host", default=os.getenv("DB_HOST"))
    p.add_argument("--db-port", default=int(os.getenv("DB_PORT", "5432")), type=int)
    p.add_argument("--db-name", default=os.getenv("DB_NAME"))
    p.add_argument("--db-user", default=os.getenv("DB_USER"))
    p.add_argument("--db-password", default=os.getenv("DB_PASSWORD"))
    
    args = p.parse_args()
    
    # Validate DB params
    required_db_params = ["db_host", "db_name", "db_user", "db_password"]
    for param in required_db_params:
        if getattr(args, param) in (None, ""):
            logger.error("Thiếu tham số DB: %s", param.upper())
            sys.exit(1)
    
    return args


def main():
    args = parse_args()
    
    try:
        # 1. Kết nối Redis
        logger.info("Đang kết nối đến Redis...")
        redis_client, redis_prefix = get_redis_connection(
            args.redis_host,
            args.redis_port,
            args.redis_db,
            args.redis_password,
            args.redis_prefix
        )
        
        # 2. Đọc recommendations từ Redis
        logger.info("Đang đọc recommendations từ Redis cho user %d...", args.user_id)
        recommendations = get_recommendations_from_redis(
            redis_client,
            redis_prefix,
            args.user_id,
            args.limit
        )
        
        if not recommendations:
            logger.warning("Không có recommendations nào cho user %d", args.user_id)
            return
        
        # 3. Lấy product IDs
        product_ids = [pid for pid, _ in recommendations]
        
        # 4. Kết nối database và lấy thông tin sản phẩm
        logger.info("Đang kết nối đến database...")
        db_conn = get_db_connection(
            args.db_host,
            args.db_port,
            args.db_name,
            args.db_user,
            args.db_password
        )
        
        logger.info("Đang lấy thông tin sản phẩm từ database...")
        products_info = get_products_info(db_conn, product_ids)
        db_conn.close()
        
        # 5. Hiển thị kết quả
        display_recommendations(args.user_id, recommendations, products_info)
        
        logger.info("Hoàn thành!")
        
    except redis.RedisError as e:
        logger.error("Lỗi kết nối Redis: %s", e)
        sys.exit(1)
    except psycopg2.Error as e:
        logger.error("Lỗi kết nối database: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Lỗi không xác định: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

