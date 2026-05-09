# Paperdesk

Paperdesk 是一个面向科研场景的论文工作台，用于整理文献、发起研究任务并生成研究报告。

## 目录结构

- `backend/`：FastAPI 后端接口与研究任务编排逻辑
- `frontend/`：Vue 3 前端界面

## 本地启动

### 启动后端

```bash
cd backend
uv sync
uv run uvicorn app.api.main:app --reload --port 8000
```

### 启动前端

```bash
cd frontend
npm install
npm run dev
```
