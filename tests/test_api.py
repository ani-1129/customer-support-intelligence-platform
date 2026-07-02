import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_token_auth(client):
    # Attempt login with correct password
    response = client.post(
        "/api/token",
        data={"username": "agent_john", "password": "password123"}
    )
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

    # Attempt login with bad password
    bad_response = client.post(
        "/api/token",
        data={"username": "agent_john", "password": "wrongpassword"}
    )
    assert bad_response.status_code == 401


def test_secured_endpoints_without_auth(client):
    # Ingest endpoint should deny anonymous access
    response = client.post("/api/ingest", json={"text": "hello"})
    assert response.status_code == 401


def test_full_pipeline_flow(client):
    # 1. Login
    login_res = client.post(
        "/api/token",
        data={"username": "agent_john", "password": "password123"}
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Ingest Ticket
    ticket_text = "Customer: I need a refund on my invoice. My email is user@test.com. The platform is payment gateway."
    ingest_res = client.post(
        "/api/ingest",
        json={"text": ticket_text},
        headers=headers
    )
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["status"] == "success"
    doc_id = ingest_data["doc_id"]
    assert doc_id is not None
    assert "[EMAIL]" in ingest_data["pii_masked_text"]

    # 3. Query RAG System
    query_res = client.post(
        "/api/query",
        json={"ticket_id": doc_id},
        headers=headers
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert "query_id" in query_data
    assert "summary" in query_data
    assert "metadata" in query_data
    assert "recommendations" in query_data
    assert "retrieved_tickets" in query_data
    
    # 4. Submit Agent Feedback
    feedback_res = client.post(
        "/api/feedback",
        json={
            "ticket_id": doc_id,
            "query_id": query_data["query_id"],
            "summary_rating": 4,
            "intent_rating": 5,
            "recommendation_rating": 4,
            "comments": "Good recommendations, minor edit in summary",
            "corrected_summary": "Customer requests double charge refund for invoice payment."
        },
        headers=headers
    )
    assert feedback_res.status_code == 200
    assert feedback_res.json()["status"] == "success"


def test_admin_metrics(client):
    # Login as normal agent
    agent_login = client.post(
        "/api/token",
        data={"username": "agent_john", "password": "password123"}
    )
    agent_token = agent_login.json()["access_token"]
    
    # Inquire metrics as agent (should fail with 403 Forbidden)
    res_agent = client.get("/api/metrics", headers={"Authorization": f"Bearer {agent_token}"})
    assert res_agent.status_code == 403

    # Login as admin
    admin_login = client.post(
        "/api/token",
        data={"username": "admin_boss", "password": "password123"}
    )
    admin_token = admin_login.json()["access_token"]
    
    # Inquire metrics as admin (should succeed)
    res_admin = client.get("/api/metrics", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_admin.status_code == 200
    metrics = res_admin.json()
    assert "total_tickets" in metrics
    assert "total_feedbacks" in metrics
    assert "average_ratings" in metrics
    assert "intent_distribution" in metrics
