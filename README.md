# Customer Support Intelligence Platform 🤖

> RAG-driven Support Ticket Summarisation, Metadata Extraction, and Agent Next-Action Recommender.

---

## 🚀 Quick Start (Local Development)

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
# Navigate to the project root
cd "D:\new project"

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root or export variables:
```bash
# Provide to disable Mock mode and use production models
export OPENAI_API_KEY="your-openai-api-key"

# Database and MLflow configuration (optional)
export QDRANT_HOST=":memory:"
export DB_URL="sqlite:///D:/new project/platform.db"
export MLFLOW_TRACKING_URI="sqlite:///D:/new project/mlflow.db"
```

### 3. Run FastAPI Backend
```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```
You can access the API Swagger docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Run Streamlit UI Dashboard
In a separate terminal (with virtual environment activated):
```bash
streamlit run ui/app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🐳 Docker Deployment

To spin up the entire microservices stack (Qdrant Vector DB, FastAPI backend, and Streamlit agent portal) using Docker Compose:
```bash
cd "D:\new project\infra"
docker-compose up --build
```
- FastAPI Server: [http://localhost:8000](http://localhost:8000)
- Streamlit Portal: [http://localhost:8501](http://localhost:8501)
- Qdrant UI Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

## 📈 Running Evaluations & MLflow Tracking

### 1. Run Pipeline Evaluations
To run the automated ROUGE, nDCG, MRR and intent extraction accuracy tests, and output plotly charts:
```bash
python notebooks/evaluate.py
```
This writes reports and standalone HTML charts to the [notebooks/plots](file:///D:/new%20project/notebooks/plots) folder.

### 2. Run MLflow Tracking Dashboard
Start the local MLflow dashboard to review logged runs and prompt versions:
```bash
mlflow server --backend-store-uri sqlite:///D:/new_project/mlflow.db --port 5000
```
Open [http://localhost:5000](http://localhost:5000) to inspect parameters, metrics, and prompts.

---

## 🔌 API Endpoints Usage Examples

### 1. Obtain Auth Token
```bash
curl -X POST "http://localhost:8000/api/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=agent_john&password=password123"
```
Returns a JWT `access_token` string.

### 2. Ingest Ticket
```bash
curl -X POST "http://localhost:8000/api/ingest" \
     -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"text": "Customer: Reset my credential. Email bob@domain.com."}'
```

### 3. Query RAG Recommendations
```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"text": "Customer: Reset my credential. Email bob@domain.com.", "top_k": 3}'
```

### 4. Submit Agent Feedback
```bash
curl -X POST "http://localhost:8000/api/feedback" \
     -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"ticket_id": "<INGEST_DOC_ID>", "summary_rating": 5, "intent_rating": 4, "corrected_summary": "User is looking to reset credentials"}'
```
