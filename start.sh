#!/usr/bin/env bash
# ============================================================
# Quant Agent — One-command launcher
# Usage: bash start.sh
# ============================================================
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# ── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Quant Agent — Starting Services       ${NC}"
echo -e "${CYAN}========================================${NC}"

# ── Python virtualenv ────────────────────────────────────────
if [ ! -d "$BACKEND/.venv" ]; then
  echo -e "${YELLOW}Creating Python venv...${NC}"
  python3 -m venv "$BACKEND/.venv"
fi

source "$BACKEND/.venv/bin/activate" 2>/dev/null || source "$BACKEND/.venv/Scripts/activate"

echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install -q -r "$BACKEND/requirements.txt"

# ── Frontend deps ────────────────────────────────────────────
if [ ! -d "$FRONTEND/node_modules" ]; then
  echo -e "${YELLOW}Installing Node dependencies...${NC}"
  cd "$FRONTEND" && npm install && cd "$ROOT"
fi

# ── Start backend ────────────────────────────────────────────
echo -e "${GREEN}Starting FastAPI backend on http://127.0.0.1:8000 ...${NC}"
cd "$BACKEND"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd "$ROOT"

# ── Start frontend ───────────────────────────────────────────
echo -e "${GREEN}Starting Vite frontend on http://localhost:5173 ...${NC}"
cd "$FRONTEND"
npm run dev &
FRONTEND_PID=$!
cd "$ROOT"

echo -e "${CYAN}----------------------------------------${NC}"
echo -e "  Backend  → http://127.0.0.1:8000"
echo -e "  Frontend → http://localhost:5173"
echo -e "  API Docs → http://127.0.0.1:8000/docs"
echo -e "${CYAN}----------------------------------------${NC}"
echo -e "Press Ctrl+C to stop all services."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo -e '${YELLOW}Stopped.${NC}'" INT TERM
wait
