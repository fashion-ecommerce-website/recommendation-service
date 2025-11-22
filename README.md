# AI Recommendation Service

Service cung cấp API cho hệ thống recommendation sử dụng IALS (Implicit Alternating Least Squares).

## 🚀 Quick Start

### Option 1: Docker (Khuyến nghị)

Nếu bạn có Docker, đây là cách nhanh nhất:

**Lưu ý**: Docker chỉ chạy recommendation service. PostgreSQL và Redis cần chạy bên ngoài.

```bash
# Đảm bảo PostgreSQL và Redis đang chạy
cp env.example .env
# Chỉnh sửa .env (DB_HOST, REDIS_HOST = localhost hoặc host.docker.internal)
docker-compose up -d
docker-compose exec recommendation-service python train_ials.py
```

Xem [DOCKER_SETUP.md](./DOCKER_SETUP.md) để biết chi tiết.

### Option 2: Manual Setup

Xem [QUICK_START.md](./QUICK_START.md) để có hướng dẫn setup nhanh trên máy mới.

## 📋 Yêu Cầu

- **Python**: 3.8+ (khuyến nghị Python 3.9-3.11)
- **PostgreSQL**: Để train model (cần có bảng `interactions`)
- **Redis**: Để cache recommendations

## 📦 Cài Đặt

### Bước 1: Clone Code

```bash
git clone <repository-url>
cd recommendation-service
```

### Bước 2: Tạo Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements_api.txt
```

**Lưu ý quan trọng**: 
- File `requirements_api.txt` đã được cấu hình với `numpy>=1.26.0,<2.0.0` để tương thích với `implicit==0.7.0`
- Nếu gặp lỗi NumPy 2.x, chạy: `pip install "numpy>=1.26.0,<2.0.0"`

### Bước 4: Cấu Hình Environment Variables

```bash
# Copy file template
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

Sau đó chỉnh sửa file `.env` với thông tin của bạn:

```env
# Database (bắt buộc cho training)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password

# Redis (bắt buộc)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Model (relative path, tự động tạo)
MODEL_DIR=model

# API
API_PORT=5000
LOG_LEVEL=INFO
```

## 🎯 Sử Dụng

### Train Model (Lần Đầu)

Trước khi chạy API service, cần train model:

```bash
python train_ials.py
```

Script sẽ:
- ✅ Đọc dữ liệu từ PostgreSQL (bảng `interactions`)
- ✅ Train model IALS
- ✅ Lưu model vào `model/ials_model_and_meta.pkl` (relative path)
- ✅ Ghi recommendations vào Redis

### Chạy API Service

```bash
python api_service.py
```

Service sẽ chạy tại: `http://localhost:5000`

**Lưu ý**: Đảm bảo đã train model trước, hoặc có file model sẵn trong thư mục `model/`

### Retrain Model (Qua API)

```bash
curl -X POST http://localhost:5000/admin/retrain-model
```

Training sẽ chạy trong background thread, không block API.

## 📡 API Endpoints

### 1. Health Check
- **GET** `/health`
- Trả về status của service
- Response: `{"status": "ok", "service": "ai-recommendation"}`

### 2. Retrain Model (Admin)
- **POST** `/admin/retrain-model`
- Trigger retrain model trong background
- Response: `{"status": "ok", "message": "Training started in background"}`

### 3. User Recommendations
- **GET** `/recs/user/{user_id}`
- Lấy gợi ý cá nhân hóa đã tính trước từ Redis
- Response: `[1, 5, 10, 15, ...]` (array of product IDs)
- Nếu không có: `404 Not Found`

### 4. Similar Items
- **GET** `/recs/item/{item_id}?limit=10`
- Tính toán "nóng" similar items từ model
- Response: `[2, 3, 7, 9, ...]` (array of product IDs)

### 5. Most Popular
- **GET** `/recs/global/most-popular?limit=10`
- Lấy sản phẩm phổ biến nhất từ Redis
- Fallback cuối cùng cho user "lạnh"
- Response: `[100, 101, 102, ...]` (array of product IDs)

## 🗄️ Redis Keys

- `rec:user:{user_id}` - Personalized recommendations (ZSET, score = rank)
- `rec:global:most-popular` - Most popular items (ZSET, score = popularity)

## 📁 Cấu Trúc Thư Mục

```
recommendation-service/
├── api_service.py          # API service chính
├── train_ials.py           # Script training model
├── requirements_api.txt    # Python dependencies
├── env.example            # Template cho .env
├── .env                   # Environment variables (tạo từ env.example)
├── README.md              # File này
├── QUICK_START.md         # Hướng dẫn setup nhanh
├── model/                 # Thư mục chứa model (tự động tạo)
│   └── ials_model_and_meta.pkl
└── venv/                  # Virtual environment (không commit)
```

## 🔧 Troubleshooting

### Lỗi: NumPy 2.x không tương thích

```bash
pip uninstall numpy -y
pip install "numpy>=1.26.0,<2.0.0"
```

Xem thêm: [FIX_NUMPY_ERROR.md](./FIX_NUMPY_ERROR.md) (nếu có)

### Lỗi: Model file not found

Chạy training trước:
```bash
python train_ials.py
```

### Lỗi: Cannot connect to Redis

Đảm bảo Redis đang chạy:
```bash
# Windows: Kiểm tra Redis service
# Linux: sudo systemctl status redis
# Mac: brew services list
```

### Lỗi: Cannot connect to PostgreSQL

Kiểm tra:
- PostgreSQL đang chạy
- Database name, user, password đúng trong `.env`
- Bảng `interactions` đã tồn tại và có dữ liệu

## ⚙️ Configuration

Tất cả cấu hình đều thông qua file `.env` hoặc environment variables. Xem `env.example` để biết các biến có sẵn.

### Training Parameters

Có thể override bằng CLI args hoặc environment variables:

```bash
# Ví dụ: Train với parameters khác
python train_ials.py --factors 128 --iterations 30 --top-n 20
```

## 🔒 Security Notes

- ⚠️ **KHÔNG commit** file `.env` vào git (chứa thông tin nhạy cảm)
- ⚠️ **KHÔNG commit** thư mục `venv/` vào git
- ✅ File `model/ials_model_and_meta.pkl` có thể rất lớn, cân nhắc khi commit
- ✅ Tất cả paths đều là relative paths, code sẽ chạy được trên mọi máy

## 📝 Notes

- Service sử dụng **relative paths** cho tất cả files, có thể chạy trên mọi máy
- Model được lưu trong thư mục `model/` (relative path từ script location)
- Training chạy trong background thread khi gọi qua API
- Recommendations được cache trong Redis với TTL mặc định 7 ngày

## 🚀 Production Deployment

Xem thêm các file hướng dẫn deployment nếu có (SETUP_GUIDE.md, etc.)

