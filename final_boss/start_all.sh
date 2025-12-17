#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

PY_DIR="$ROOT_DIR"
BACKEND_DIR="$ROOT_DIR/web/backend"
FRONTEND_DIR="$ROOT_DIR/web/frontend"

echo "==> [1/3] Start Python AI (FastAPI) on :8000"
cd "$PY_DIR"
if [ ! -d "venv" ]; then
  echo "Không thấy venv/. Tạo venv..."
  python3 -m venv venv
fi
source venv/bin/activate
pip -q install -U pip >/dev/null
pip -q install fastapi uvicorn scikit-learn pandas joblib >/dev/null

uvicorn ml_api:app --host 0.0.0.0 --port 8000 > "$ROOT_DIR/.log_python.txt" 2>&1 &
PY_PID=$!
echo "   Python PID: $PY_PID (log: .log_python.txt)"

echo "==> [2/3] Start Node backend on :3000"
cd "$BACKEND_DIR"
npm install >/dev/null 2>&1 || true
npm run dev > "$ROOT_DIR/.log_backend.txt" 2>&1 &
BE_PID=$!
echo "   Backend PID: $BE_PID (log: .log_backend.txt)"

echo "==> [3/3] Start Vite frontend on :5173"
cd "$FRONTEND_DIR"
npm install >/dev/null 2>&1 || true
npm run dev -- --host 0.0.0.0 > "$ROOT_DIR/.log_frontend.txt" 2>&1 &
FE_PID=$!
echo "   Frontend PID: $FE_PID (log: .log_frontend.txt)"

echo ""
echo "✅ All started!"
echo "   - AI API:      http://localhost:8000/docs"
echo "   - Backend API: http://localhost:3000"
echo "   - Frontend:    http://localhost:5173"
echo ""
echo "Nhấn Ctrl+C để dừng toàn bộ."

cleanup() {
  echo ""
  echo "Stopping..."
  kill $FE_PID >/dev/null 2>&1 || true
  kill $BE_PID >/dev/null 2>&1 || true
  kill $PY_PID >/dev/null 2>&1 || true
  echo "Done."
}
trap cleanup INT TERM

wait
