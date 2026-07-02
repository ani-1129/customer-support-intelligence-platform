# 🤖 Customer Support Intelligence Platform

> **RAG-driven Ticket Summarisation, Intent Extraction, and Agent Recommendation System**

---

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://support-intelligence-platform.streamlit.app/)
[![FastAPI Backend](https://img.shields.io/badge/⚡%20FastAPI%20Backend-Live%20on%20Render-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://customer-support-api-4xns.onrender.com)
[![API Docs](https://img.shields.io/badge/📖%20API%20Docs-Swagger%20UI-blue?style=for-the-badge&logo=swagger&logoColor=white)](https://customer-support-api-4xns.onrender.com/docs)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC143C?style=flat-square)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 🌐 Live Links

| Service | URL |
|---|---|
| 🎯 **Agent Dashboard (Streamlit UI)** | [https://support-intelligence-platform.streamlit.app/](https://support-intelligence-platform.streamlit.app/) |
| ⚡ **FastAPI Backend** | [https://customer-support-api-4xns.onrender.com](https://customer-support-api-4xns.onrender.com) |
| 📖 **Interactive API Docs (Swagger)** | [https://customer-support-api-4xns.onrender.com/docs](https://customer-support-api-4xns.onrender.com/docs) |
| 💚 **Health Check** | [https://customer-support-api-4xns.onrender.com/api/health](https://customer-support-api-4xns.onrender.com/api/health) |

> ⚠️ **Note:** The backend is hosted on Render's free tier and may take **~30 seconds** to wake up after inactivity. The Streamlit app will show a loading spinner while the API boots up.

---

## 📌 What This Project Does

Given a raw customer support interaction (chat transcript, email, or voice transcript), the platform:

1. 🧹 **Cleans & Anonymises** the ticket — strips PII (emails, phones, SSNs, credit cards)
2. 📝 **Summarises** the issue in 2–3 concise sentences an agent can read instantly
3. 🏷️ **Extracts metadata** — intent, products mentioned, SLA priority, sentiment, and named entities
4. 🔍 **Retrieves** 3–5 semantically similar past tickets from a vector database (RAG)
5. ✅ **Recommends** next-actions, Knowledge Base articles, and a step-by-step agent runbook
6. 🔄 **Learns** from human agent feedback (Likert ratings) to improve over time

---

## 🏗️ Architecture

```
Raw Transcript / Email
        ↓
┌─────────────────────┐
│  Ingestion & Cleaning│  ← PII Masking, Speaker Splitting
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Embedding Generator │  ← Mock / SentenceTransformers / OpenAI
└─────────────────────┘
        ↓
┌─────────────────────┐     ┌────────────────────┐
│   Qdrant Vector DB   │ ←→  │  RAG Query Engine  │
└─────────────────────┘     └────────────────────┘
                                      ↓
                           ┌─────────────────────┐
                           │  Cross-Encoder       │  ← TF-IDF / Transformer Reranker
                           │  Reranker            │
                           └─────────────────────┘
                                      ↓
                           ┌─────────────────────┐
                           │  LLM Orchestrator    │  ← Mock / OpenAI
                           │  • Summariser        │
                           │  • Intent Extractor  │
                           │  • Recommender       │
                           └─────────────────────┘
                                      ↓
                           ┌─────────────────────┐
                           │  Agent Dashboard     │  ← Streamlit UI
                           │  + Feedback Loop     │
                           └─────────────────────┘
                                      ↓
                              SQLite / PostgreSQL
                              (Feedback & Logs)
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone the Repository
```bash
git clone https://github.com/ani-1129/customer-support-intelligence-platform.git
cd customer-support-intelligence-platform
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional)
Create a `.env` file or export these variables:
```bash
# Leave blank to run in offline Mock mode (no API key needed)
OPENAI_API_KEY=your-openai-api-key

# Database & Vector DB (defaults work out of the box)
QDRANT_HOST=:memory:
DB_URL=sqlite:///./platform.db
SECRET_KEY=your-secret-key-here
```

### 4. Start the FastAPI Backend
```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```
📖 API docs available at: http://127.0.0.1:8000/docs

### 5. Start the Streamlit Dashboard
Open a second terminal:
```bash
streamlit run ui/app.py --server.port 8501
```
🌐 Dashboard opens at: http://localhost:8501

---

## 🐳 Docker Deployment

Spin up the full stack (Qdrant + FastAPI + Streamlit) with one command:
```bash
cd infra
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---

## 🔌 API Endpoints

### Authentication
```bash
# Get JWT Token (Agent)
curl -X POST "https://customer-support-api-4xns.onrender.com/api/token" \
     -d "username=agent_john&password=password123"

# Get JWT Token (Admin)
curl -X POST "https://customer-support-api-4xns.onrender.com/api/token" \
     -d "username=admin_boss&password=password123"
```

### Ingest a Ticket
```bash
curl -X POST "https://customer-support-api-4xns.onrender.com/api/ingest" \
     -H "Authorization: Bearer <YOUR_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"text": "Customer: I was charged twice for my subscription. My email is bob@test.com."}'
```

### Query the RAG Pipeline
```bash
curl -X POST "https://customer-support-api-4xns.onrender.com/api/query" \
     -H "Authorization: Bearer <YOUR_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"text": "Customer: I cannot log in, the reset link gives a 404 error.", "top_k": 3}'
```

### Submit Agent Feedback
```bash
curl -X POST "https://customer-support-api-4xns.onrender.com/api/feedback" \
     -H "Authorization: Bearer <YOUR_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"ticket_id": "<DOC_ID>", "summary_rating": 5, "intent_rating": 4, "recommendation_rating": 5}'
```

### Health Check
```bash
curl "https://customer-support-api-4xns.onrender.com/api/health"
```

---

## 🎮 Demo Credentials

| Role | Username | Password | Access |
|---|---|---|---|
| 🧑‍💼 Support Agent | `agent_john` | `password123` | Ingest, Query, Feedback |
| 👑 Supervisor/Admin | `admin_boss` | `password123` | All + Metrics Dashboard |

---

## 📊 Evaluation Results

| Metric | Baseline | With Reranker |
|---|---|---|
| **MRR (Retrieval)** | `0.800` | `1.000` ✅ |
| **nDCG@3** | `0.584` | `0.890` ✅ |
| **ROUGE-1 (Summary)** | — | `0.436` |
| **Intent F1-Score** | — | `0.800` |

---

## 🗂️ Project Structure

```
customer-support-intelligence-platform/
├── src/                    # Core pipeline modules
│   ├── config.py           # Environment config
│   ├── ingestion.py        # Cleaning, PII masking, speaker split
│   ├── embeddings.py       # Mock / SentenceTransformers / OpenAI
│   ├── vector_db.py        # Qdrant wrapper
│   ├── retriever.py        # RAG retrieval
│   ├── reranker.py         # TF-IDF & CrossEncoder reranker
│   ├── llm.py              # Mock & OpenAI LLM handlers
│   ├── summariser.py       # Summarisation prompts
│   ├── extractor.py        # Intent/entity extraction
│   ├── recommender.py      # Next-actions & KB articles
│   ├── database.py         # SQLAlchemy schema
│   └── mlflow_utils.py     # Experiment tracking
├── api/                    # FastAPI backend
│   ├── app.py              # Endpoints: /ingest /query /feedback /metrics
│   └── auth.py             # OAuth2 JWT + RBAC + rate limiting
├── ui/                     # Streamlit dashboard
│   └── app.py              # Agent sandbox + supervisor analytics
├── notebooks/              # Evaluation scripts
│   └── evaluate.py         # ROUGE, nDCG, MRR, plots
├── tests/                  # Pytest suite (11 tests)
│   ├── test_pipeline.py
│   └── test_api.py
├── infra/                  # Docker assets
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── docker-compose.yml
├── docs/
│   └── report.md           # 5-page technical report
├── render.yaml             # Render auto-deploy config
├── Procfile                # Render start command
├── requirements.txt
└── README.md
```

---

## 🛡️ Security Features

- 🔐 **OAuth2 JWT Authentication** on all API endpoints
- 🧑‍💼 **Role-Based Access Control** (Agent vs Admin)
- 🚦 **Rate Limiting** — sliding window per user token
- 🔒 **PII Redaction** — emails, phones, SSNs, credit cards masked before indexing
- 🔑 **Secret Key** via environment variable (never hardcoded)

---

## 📈 MLflow Experiment Tracking

```bash
# Run locally to track prompt versions and metrics
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
# Then open: http://localhost:5000
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

**Built with ❤️ using FastAPI, Streamlit, Qdrant, and LangChain**

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20the%20Live%20Demo-Click%20Here-FF4B4B?style=for-the-badge)](https://support-intelligence-platform.streamlit.app/)

</div>
