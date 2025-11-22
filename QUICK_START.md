# 🚀 Quick Start Guide

Hướng dẫn nhanh để setup và chạy AI Recommendation Service trên máy mới (chưa setup gì).

## 🐳 Option 1: Docker (Khuyến nghị - Dễ nhất)

Nếu bạn có Docker, đây là cách nhanh nhất:

**Lưu ý**: Docker chỉ chạy recommendation service. PostgreSQL và Redis cần chạy bên ngoài.

```bash
# 1. Đảm bảo PostgreSQL và Redis đang chạy trên host
# Kiểm tra: redis-cli ping

# 2. Chỉnh sửa docker-compose.yml hoặc tạo .env
#    Cách 1: Sửa trực tiếp trong docker-compose.yml (khuyến nghị)
#    Cách 2: Tạo .env và uncomment volume mount trong docker-compose.yml
#    QUAN TRỌNG: Nếu dùng .env, phải set:
#      - DB_HOST=host.docker.internal (KHÔNG phải localhost)
#      - REDIS_HOST=host.docker.internal (KHÔNG phải localhost)

# 3. Start service
docker-compose up -d

# 5. Train model (lần đầu - có thể train từ host hoặc container)
docker-compose exec recommendation-service python train_ials.py
# Hoặc từ host: python train_ials.py

# 6. Test
curl http://localhost:5000/health
```

Xem [DOCKER_SETUP.md](./DOCKER_SETUP.md) để biết chi tiết.

---

## 💻 Option 2: Manual Setup (Không dùng Docker)

## ⚡ 5 Bước Setup

### Bước 1: Clone Code

```bash
git clone <repository-url>
cd recommendation-service
```

### Bước 2: Tạo Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv venv
venv\Scripts\activate.bat

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements_api.txt
```

**Kiểm tra**: Nếu gặp lỗi về NumPy, chạy:
```bash
pip install "numpy>=1.26.0,<2.0.0"
```

### Bước 4: Cấu Hình Environment

```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

Mở file `.env` và điền thông tin:

```env
# Database (bắt buộc)
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

# Model (relative path - không cần sửa)
MODEL_DIR=model

# API (có thể giữ mặc định)
API_PORT=5000
LOG_LEVEL=INFO
```

### Bước 5: Train Model & Chạy Service

```bash
# 1. Train model (lần đầu)
python train_ials.py

# 2. Chạy API service
python api_service.py
```

## ✅ Kiểm Tra

Mở browser hoặc terminal:

```bash
# Health check
curl http://localhost:5000/health

# Hoặc mở: http://localhost:5000/health
```

Nếu thấy `{"status": "ok", "service": "ai-recommendation"}` → ✅ Thành công!

## 🎯 Test API

```bash
# Lấy recommendations cho user_id = 1
curl http://localhost:5000/recs/user/1

# Lấy similar items cho item_id = 100
curl http://localhost:5000/recs/item/100?limit=10

# Lấy most popular items
curl http://localhost:5000/recs/global/most-popular?limit=10
```

## ❌ Troubleshooting

### Lỗi: "No module named 'numpy'"
```bash
pip install -r requirements_api.txt
```

### Lỗi: "Model file not found"
```bash
# Chạy training trước
python train_ials.py
```

### Lỗi: "Failed to connect to Redis"
- Kiểm tra Redis đang chạy: `redis-cli ping`
- Kiểm tra `REDIS_HOST` và `REDIS_PORT` trong `.env`

### Lỗi: "Cannot connect to PostgreSQL"
- Kiểm tra PostgreSQL đang chạy
- Kiểm tra thông tin DB trong `.env`
- Đảm bảo bảng `interactions` đã tồn tại

### Lỗi: NumPy 2.x không tương thích
```bash
pip uninstall numpy -y
pip install "numpy>=1.26.0,<2.0.0"
```

## 📋 Checklist Setup

- [ ] Python 3.8+ đã cài đặt
- [ ] PostgreSQL đang chạy và có dữ liệu
- [ ] Redis đang chạy
- [ ] Virtual environment đã tạo và activate
- [ ] Dependencies đã cài đặt (`pip install -r requirements_api.txt`)
- [ ] File `.env` đã tạo và cấu hình đúng
- [ ] Model đã train (`python train_ials.py`)
- [ ] API service đang chạy (`python api_service.py`)

## 🎉 Hoàn Thành!

Nếu tất cả đều OK, bạn đã sẵn sàng sử dụng API recommendation service!

Xem [README.md](./README.md) để biết thêm chi tiết về các endpoints và cấu hình.

