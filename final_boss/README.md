# Hướng dẫn chạy project (Local – WSL)

## Yêu cầu môi trường
- Ubuntu / WSL
- Node.js >= 18
- Python >= 3.9
- MySQL 8+

---

## 1. Khởi động MySQL
```bash

sudo service mysql start
2. Tạo database (chỉ cần lần đầu)
Cách 1 (khuyến nghị)

mysql -u appuser -p < full_database.sql
Hoặc

mysql -u appuser -p KiThuatDuLieu < full_database.sql
Sau đó chạy thêm:

mysql -u appuser -p KiThuatDuLieu < web/backend/sql/history.sql

3. Backend AI (Python – FastAPI)
Tại thư mục root của project:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Chạy AI server:


uvicorn ml_api:app --host 0.0.0.0 --port 8000
Test nhanh:


http://localhost:8000/docs

4. Backend API (Node.js)
Mở terminal mới:

cd web/backend
npm install
npm run dev

5. Frontend (Vite + React)
Mở terminal mới:


cd web/frontend
npm install
npm run dev

6. Cấu hình môi trường (.env)
Tạo file .env trong web/backend/:


DB_HOST=localhost
DB_USER=appuser
DB_PASSWORD=your_password
DB_NAME=KiThuatDuLieu
JWT_SECRET=your_secret_key


------------------------------
username: giau 
password: 123456

username: rumeodinhau
password: 123

username: rua 
password: 123

username: phu
password: 123


-----------------------------
# # 1. Mở WSL
# sudo service mysql start

# # 2. (chỉ cần lần đầu)
# - mysql -u appuser -p < full_database.sql                    <----dùng cái này đi bro
# hoặc: mysql -u appuser -p KiThuatDuLieu < full_database.sql
# chạy tiếp:
# mysql -u appuser -p KiThuatDuLieu < full_database.sql

# ------------------------------------
# Tại thư mục root của project:

# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# --------------------
# Sau đó chạy backend AI(chỗ có phần venv nha):

# uvicorn ml_api:app --host 0.0.0.0 --port 8000
# test nhanh vào: http://localhost:8000/docs 
# thấy ok là được
# ------------------
# Mở terminal mới, vào backend:

# cd web/backend
# npm install
# npm run dev
# --------------------
# Mở terminal mới, vào frontend:

# cd web/frontend
# npm install
# npm run dev


