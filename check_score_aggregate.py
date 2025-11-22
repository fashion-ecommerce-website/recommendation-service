import os
import sys
import argparse
import pickle
import logging
import numpy as np
import psycopg2
from dotenv import load_dotenv

# Tải file .env
load_dotenv()

# Cấu hình logging
logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_model_data(model_path: str):
    """Tải file .pkl chứa model và mappings."""
    if not os.path.exists(model_path):
        logger.error(f"Không tìm thấy file model: {model_path}")
        sys.exit(1)
    try:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        logger.info(f"Tải model thành công từ {model_path}")
        return payload
    except Exception as e:
        logger.error(f"Lỗi khi tải model: {e}")
        sys.exit(1)


def get_predicted_score(model_data, real_user_id: int, real_product_id: int) -> float | None:
    """
    Tính điểm dự đoán cho một cặp (user, product) cụ thể.
    Trả về None nếu user hoặc product không có trong model.
    """
    model = model_data["model"]
    user_map = model_data["user_ids_map"]
    product_map = model_data["product_ids_map"]

    if real_user_id not in user_map:
        logger.warning(f"User ID {real_user_id} không có trong dữ liệu train.")
        return None
    if real_product_id not in product_map:
        # Điều này là bình thường, nếu sản phẩm đó không có tương tác nào
        # logger.warning(f"Product ID {real_product_id} không có trong dữ liệu train.")
        return None

    internal_user_idx = user_map[real_user_id]
    internal_product_idx = product_map[real_product_id]

    user_vector = model.user_factors[internal_user_idx]
    product_vector = model.item_factors[internal_product_idx]

    score = np.dot(user_vector, product_vector)
    return float(score)


def get_db_conn():
    """Tạo kết nối DB từ file .env."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=5,
        )
        return conn
    except Exception as e:
        logger.error(f"Lỗi kết nối DB: {e}")
        sys.exit(1)


def fetch_product_groups(conn) -> dict[int, str]:
    """
    Lấy "gu" của TẤT CẢ sản phẩm.
    Sử dụng y hệt logic CASE từ script sinh data của bạn.
    """
    sql = """
    SELECT
        p.id AS product_id,
        CASE
            WHEN p.title ILIKE 'Túi%' THEN 'BAG'
            WHEN p.title ILIKE 'Nón bóng chày%' THEN 'CAP'
            WHEN p.title ILIKE 'Nón bucket%' THEN 'BUCKET'
            WHEN p.title ILIKE 'Quần short%' OR p.title ILIKE 'Quần jogger%' THEN 'BOTTOM'
            WHEN p.title ILIKE 'Áo sweatshirt%' OR p.title ILIKE 'Áo sơ mi%'
              OR p.title ILIKE 'Áo polo%' OR p.title ILIKE 'Áo thun%' THEN 'TOP'
            ELSE 'MISC'
        END AS g
    FROM products p
    WHERE p.is_active = TRUE;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    # Trả về dict: {product_id: group_name}
    return {row[0]: row[1] for row in rows}


def fetch_user_history(conn, real_user_id: int) -> set[int]:
    """Lấy set các product_id mà user ĐÃ tương tác."""
    sql = "SELECT DISTINCT product_id FROM interactions WHERE user_id = %s;"
    with conn.cursor() as cur:
        cur.execute(sql, (real_user_id,))
        rows = cur.fetchall()
    # Trả về set: {pid1, pid2, ...}
    return {row[0] for row in rows}


def calculate_average_scores(scores: list[float]) -> float:
    """Tính trung bình an toàn."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra điểm dự đoán TRUNG BÌNH của IALS cho 2 nhóm 'gu'.")
    parser.add_argument("-u", "--user-id", type=int, required=True, help="ID thật của User (ví dụ: 3)")
    parser.add_argument("-a", "--group-a", type=str, required=True, help="Tên 'gu' thứ nhất (ví dụ: CAP)")
    parser.add_argument("-b", "--group-b", type=str, required=True, help="Tên 'gu' thứ hai (ví dụ: BOTTOM)")
    parser.add_argument("--model-path", default="model/ials_model_and_meta.pkl", help="Đường dẫn đến file model .pkl")

    args = parser.parse_args()

    group_a_name = args.group_a.upper()
    group_b_name = args.group_b.upper()

    # 1. Tải Model
    model_data = load_model_data(args.model_path)

    # 2. Lấy dữ liệu từ DB
    conn = get_db_conn()
    all_products_with_groups = fetch_product_groups(conn)
    user_history_set = fetch_user_history(conn, args.user_id)
    conn.close()

    # Lấy set các sản phẩm mà Model biết (đã được train)
    model_products_set = set(model_data["product_ids_map"].keys())

    logger.info(f"--- ĐANG KIỂM TRA TRUNG BÌNH CHO USER ID: {args.user_id} ---")
    logger.info(f"So sánh Gu '{group_a_name}' vs. Gu '{group_b_name}'")

    group_a_scores = []
    group_b_scores = []

    # 3. Lặp qua tất cả sản phẩm để tính điểm
    for pid, group in all_products_with_groups.items():
        # Bỏ qua nếu:
        # 1. User đã xem rồi
        # 2. Model không biết sản phẩm này (chưa được train)
        if pid in user_history_set or pid not in model_products_set:
            continue

        score = None
        if group == group_a_name:
            score = get_predicted_score(model_data, args.user_id, pid)
            if score is not None:
                group_a_scores.append(score)

        elif group == group_b_name:
            score = get_predicted_score(model_data, args.user_id, pid)
            if score is not None:
                group_b_scores.append(score)

    # 4. Tính toán và In kết quả
    avg_a = calculate_average_scores(group_a_scores)
    avg_b = calculate_average_scores(group_b_scores)

    count_a = len(group_a_scores)
    count_b = len(group_b_scores)

    print("\n--- 📊 KẾT QUẢ KIỂM TRA TRUNG BÌNH ---")
    print(f"User ID:    {args.user_id}")
    print("-" * 40)
    print(f"Nhóm '{group_a_name}':")
    print(f"   Số lượng item (chưa xem): {count_a}")
    print(f"   Điểm dự đoán TRUNG BÌNH: {avg_a:.6f}")
    print("-" * 40)
    print(f"Nhóm '{group_b_name}':")
    print(f"   Số lượng item (chưa xem): {count_b}")
    print(f"   Điểm dự đoán TRUNG BÌNH: {avg_b:.6f}")
    print("=" * 40)

    if avg_a > avg_b:
        print(f"✅ KẾT LUẬN: Model dự đoán CHÍNH XÁC.")
        print(f"   (Điểm trung bình của '{group_a_name}' cao hơn '{group_b_name}')")
    elif avg_b > avg_a:
        print(f"❌ KẾT LUẬN: Model dự đoán CHƯA CHÍNH XÁC.")
        print(f"   (Điểm trung bình của '{group_b_name}' cao hơn '{group_a_name}')")
    else:
        print("ℹ️ KẾT LUẬN: Model không phân biệt được 2 nhóm.")
    print("=" * 40)


if __name__ == "__main__":
    main()