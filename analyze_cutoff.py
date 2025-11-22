"""
Script phân tích dữ liệu và đề xuất cutoff date tối ưu cho train/test split
"""
import os
import sys
import argparse
import logging
import pandas as pd
from dotenv import load_dotenv
import psycopg2

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_db_conn(args):
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
    sql = """
        SELECT user_id, product_id, action_type, 
               CASE WHEN count IS NULL THEN 1 ELSE count END AS count, 
               created_at
        FROM interactions
        ORDER BY created_at
    """
    with get_db_conn(args) as conn:
        df = pd.read_sql_query(sql, conn)
    if df.empty:
        logger.warning("No interactions found.")
    else:
        logger.info(f"Read {len(df)} interaction rows from DB.")
    return df


def analyze_data_distribution(df: pd.DataFrame):
    """Phân tích phân bố dữ liệu theo thời gian"""
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    min_date = df['created_at'].min()
    max_date = df['created_at'].max()
    total_days = (max_date - min_date).days + 1
    
    logger.info(f"\n{'='*60}")
    logger.info("PHÂN TÍCH PHÂN BỐ DỮ LIỆU")
    logger.info(f"{'='*60}")
    logger.info(f"Ngày bắt đầu: {min_date.date()}")
    logger.info(f"Ngày kết thúc: {max_date.date()}")
    logger.info(f"Tổng số ngày: {total_days} ngày")
    logger.info(f"Tổng số interactions: {len(df):,}")
    logger.info(f"Số users: {df['user_id'].nunique():,}")
    logger.info(f"Số products: {df['product_id'].nunique():,}")
    
    # Phân bố theo ngày
    daily_stats = df.groupby(df['created_at'].dt.date).agg({
        'user_id': 'nunique',
        'product_id': 'nunique',
    })
    daily_stats['interactions'] = df.groupby(df['created_at'].dt.date).size()
    daily_stats = daily_stats.rename(columns={
        'user_id': 'unique_users',
        'product_id': 'unique_products',
    })
    
    logger.info(f"\nInteractions trung bình/ngày: {daily_stats['interactions'].mean():.1f}")
    logger.info(f"Users trung bình/ngày: {daily_stats['unique_users'].mean():.1f}")
    
    return min_date, max_date, total_days, daily_stats


def suggest_cutoffs(df: pd.DataFrame, min_date, max_date, total_days, gt_events=['PURCHASE']):
    """Đề xuất các cutoff date dựa trên phân tích"""
    df['created_at'] = pd.to_datetime(df['created_at'])
    gt_events_set = set([e.strip().upper() for e in gt_events])
    
    suggestions = []
    
    # Thử các tỷ lệ train/test phổ biến
    ratios = [
        (0.6, 0.4, "60/40 - Nhiều test data"),
        (0.7, 0.3, "70/30 - Cân bằng"),
        (0.8, 0.2, "80/20 - Nhiều train data (khuyến nghị)"),
        (0.85, 0.15, "85/15 - Tối ưu train"),
    ]
    
    logger.info(f"\n{'='*60}")
    logger.info("ĐỀ XUẤT CUTOFF DATE")
    logger.info(f"{'='*60}\n")
    
    for train_ratio, test_ratio, desc in ratios:
        cutoff_days = int(total_days * train_ratio)
        cutoff_date = min_date + pd.Timedelta(days=cutoff_days)
        
        # Tính toán train/test split
        train_df = df[df['created_at'] <= cutoff_date]
        test_df = df[df['created_at'] > cutoff_date]
        
        # Test positives (chỉ các event trong gt_events)
        test_gt = test_df[test_df['action_type'].str.upper().isin(gt_events_set)]
        test_users = test_gt['user_id'].nunique()
        
        # Users trong train
        train_users = train_df['user_id'].nunique()
        
        # Users có cả train và test
        common_users = set(train_df['user_id'].unique()) & set(test_gt['user_id'].unique())
        
        # Tính toán số liệu
        train_interactions = len(train_df)
        test_interactions = len(test_df)
        test_gt_interactions = len(test_gt)
        
        # Đánh giá chất lượng split
        quality_score = 0
        quality_notes = []
        
        # Tiêu chí 1: Test có đủ users
        if test_users >= 100:
            quality_score += 3
            quality_notes.append("✓ Test có đủ users (≥100)")
        elif test_users >= 50:
            quality_score += 2
            quality_notes.append("⚠ Test có users vừa phải (50-99)")
        else:
            quality_score += 1
            quality_notes.append("✗ Test có ít users (<50)")
        
        # Tiêu chí 2: Train có đủ dữ liệu
        if train_interactions >= 10000:
            quality_score += 3
            quality_notes.append("✓ Train có đủ interactions (≥10k)")
        elif train_interactions >= 5000:
            quality_score += 2
            quality_notes.append("⚠ Train có interactions vừa phải (5k-10k)")
        else:
            quality_score += 1
            quality_notes.append("✗ Train có ít interactions (<5k)")
        
        # Tiêu chí 3: Users có cả train và test (warm-start)
        common_ratio = len(common_users) / test_users if test_users > 0 else 0
        if common_ratio >= 0.8:
            quality_score += 2
            quality_notes.append("✓ Nhiều users có cả train và test (≥80%)")
        elif common_ratio >= 0.5:
            quality_score += 1
            quality_notes.append("⚠ Một số users chỉ có test (cold-start)")
        
        suggestions.append({
            'cutoff_date': cutoff_date.date(),
            'cutoff_date_str': cutoff_date.strftime('%Y-%m-%d'),
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'desc': desc,
            'train_users': train_users,
            'test_users': test_users,
            'common_users': len(common_users),
            'train_interactions': train_interactions,
            'test_interactions': test_interactions,
            'test_gt_interactions': test_gt_interactions,
            'quality_score': quality_score,
            'quality_notes': quality_notes
        })
        
        logger.info(f"\n{desc} (Train: {train_ratio*100:.0f}% / Test: {test_ratio*100:.0f}%)")
        logger.info(f"  Cutoff date: {cutoff_date.date()} ({cutoff_date.strftime('%Y-%m-%d')})")
        logger.info(f"  Train: {train_users:,} users, {train_interactions:,} interactions")
        logger.info(f"  Test: {test_users:,} users với GT events, {test_gt_interactions:,} GT interactions")
        logger.info(f"  Users có cả train & test: {len(common_users):,} ({common_ratio*100:.1f}%)")
        logger.info(f"  Điểm chất lượng: {quality_score}/8")
        for note in quality_notes:
            logger.info(f"    {note}")
    
    # Đề xuất cutoff tốt nhất
    best = max(suggestions, key=lambda x: x['quality_score'])
    logger.info(f"\n{'='*60}")
    logger.info(f"KHUYẾN NGHỊ: Cutoff date tốt nhất")
    logger.info(f"{'='*60}")
    logger.info(f"Cutoff: {best['cutoff_date_str']}")
    logger.info(f"Lý do: {best['desc']}, Điểm chất lượng: {best['quality_score']}/8")
    logger.info(f"\nSử dụng lệnh sau để evaluate:")
    logger.info(f"  --cutoff \"{best['cutoff_date_str']}\"")
    
    return suggestions, best


def parse_args():
    p = argparse.ArgumentParser(description="Phân tích và đề xuất cutoff date cho train/test split")
    
    p.add_argument("--db-host", default=os.getenv("DB_HOST"))
    p.add_argument("--db-port", default=os.getenv("DB_PORT", "5432"), type=int)
    p.add_argument("--db-name", default=os.getenv("DB_NAME"))
    p.add_argument("--db-user", default=os.getenv("DB_USER"))
    p.add_argument("--db-password", default=os.getenv("DB_PASSWORD"))
    p.add_argument("--gt-events", default="PURCHASE", help="Các event tính GT, phân cách bởi dấu phẩy")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Đọc dữ liệu
    df = read_interactions(args)
    if df.empty:
        logger.error("Không có dữ liệu. Dừng phân tích.")
        sys.exit(1)
    
    # Phân tích phân bố
    min_date, max_date, total_days, daily_stats = analyze_data_distribution(df)
    
    # Đề xuất cutoff
    gt_events = [e.strip() for e in args.gt_events.split(",")]
    suggestions, best = suggest_cutoffs(df, min_date, max_date, total_days, gt_events)
    
    logger.info(f"\n{'='*60}")
    logger.info("HOÀN THÀNH PHÂN TÍCH")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

