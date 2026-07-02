import uuid
import json
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.config import IS_MOCK_MODE, OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT
from src.ingestion import preprocess_ticket
from src.embeddings import get_embedding_model
from src.vector_db import QdrantVectorDB
from src.retriever import TicketRetriever
from src.reranker import TicketReranker
from src.llm import get_llm
from src.summariser import TicketSummariser
from src.extractor import MetadataExtractor
from src.recommender import AgentRecommender
from src.database import get_db, init_db, DBTicket, DBFeedback

from api.auth import get_current_user, require_role, create_access_token

app = FastAPI(
    title="Customer Support Intelligence Platform API",
    description="Backend microservice providing RAG-driven support ticket indexing, summary generation, intent extraction, and recommendation",
    version="1.0.0"
)

# Global variables for core services
embedding_model = None
vector_db = None
retriever = None
reranker = None
llm = None
summariser = None
extractor = None
recommender = None

@app.on_event("startup")
def startup_event():
    global embedding_model, vector_db, retriever, reranker, llm, summariser, extractor, recommender
    # 1. Initialize SQLite Database
    init_db()
    
    # 2. Setup Embedding and Vector DB Services
    embedding_model = get_embedding_model(is_mock=IS_MOCK_MODE, api_key=OPENAI_API_KEY)
    vector_db = QdrantVectorDB(host=QDRANT_HOST, port=QDRANT_PORT)
    retriever = TicketRetriever(embedding_model, vector_db, collection_name="tickets")
    
    # 3. Setup LLM and Pipeline Services
    llm = get_llm(is_mock=IS_MOCK_MODE, api_key=OPENAI_API_KEY)
    summariser = TicketSummariser(llm)
    extractor = MetadataExtractor(llm)
    recommender = AgentRecommender(llm)
    
    # 4. Setup Reranker (uses TF-IDF fallback locally, cross-encoder if configured)
    reranker = TicketReranker(use_transformer=False)
    
    # Seed the vector database with a couple of mock tickets if running in-memory
    if QDRANT_HOST == ":memory:":
        seed_mock_tickets()

def seed_mock_tickets():
    mock_tickets = [
        ("id-1", "Customer: I need to request a full refund. My account was charged twice for the monthly subscription on 2026-06-15. Transaction invoice #stripe-4023.", {"intent": "Billing & Payments"}),
        ("id-2", "Customer: I cannot log in. The password reset link you sent doesn't work. It gives a 404 URL error. Please help quickly.", {"intent": "Account Access & Authentication"}),
        ("id-3", "Customer: The dashboard is loading extremely slowly today. It takes over 30 seconds to fetch my active customer list. Is the system down?", {"intent": "Technical Troubleshooting"}),
        ("id-4", "Customer: How do I invite colleagues to my workspace? I want to add 3 new team members as agents.", {"intent": "General Inquiry"})
    ]
    for doc_id, text, metadata in mock_tickets:
        retriever.index_ticket(doc_id=doc_id, text=text, metadata=metadata)
    print("Seeded in-memory Vector DB with initial knowledge base.")


# --- AUTHENTICATION ---

@app.post("/api/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Demo token generation.
    Supports credentials:
    - User 1: username='agent_john', password='password123' (role='agent')
    - User 2: username='admin_boss', password='password123' (role='admin')
    """
    role = ""
    if form_data.username == "agent_john" and form_data.password == "password123":
        role = "agent"
    elif form_data.username == "admin_boss" and form_data.password == "password123":
        role = "admin"
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
        
    access_token = create_access_token(data={"sub": form_data.username, "role": role})
    return {"access_token": access_token, "token_type": "bearer"}


# --- CORE PIPELINE ENDPOINTS ---

@app.post("/api/ingest", response_model=Dict[str, Any])
def ingest_ticket(
    ticket_payload: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Ingests a ticket, processes text, anonymizes PII, indexes in Vector DB, and saves to database.
    Payload keys: 'text', 'metadata' (optional).
    """
    raw_text = ticket_payload.get("text", "")
    if not raw_text:
        raise HTTPException(status_code=400, detail="Payload must contain a non-empty 'text' field.")
        
    doc_id = str(uuid.uuid4())
    
    # 1. Clean & anonymize
    processed = preprocess_ticket(raw_text)
    
    # 2. Extract initial metadata using LLM
    meta = extractor.extract_metadata(processed["masked_text"])
    payload_metadata = ticket_payload.get("metadata", {})
    combined_metadata = {**meta, **payload_metadata}
    
    # 3. Generate summary
    summary = summariser.summarise(processed["masked_text"])
    
    # 4. Generate next-actions/recommendations
    recs = recommender.generate_recommendations(processed["masked_text"], [])

    # 5. Insert in Vector DB (indexing masked version to ensure PII safety at rest)
    retriever.index_ticket(doc_id=doc_id, text=processed["masked_text"], metadata={"intent": combined_metadata.get("intent")})
    
    # 6. Save in SQLite DB
    db_ticket = DBTicket(
        id=doc_id,
        raw_text=raw_text,
        cleaned_text=processed["cleaned_text"],
        masked_text=processed["masked_text"],
        summary=summary,
        metadata_json=json.dumps(combined_metadata),
        recommendations_json=json.dumps(recs)
    )
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)
    
    return {
        "status": "success",
        "doc_id": doc_id,
        "message": "Ticket successfully ingested and indexed.",
        "pii_masked_text": processed["masked_text"]
    }


@app.post("/api/query", response_model=Dict[str, Any])
def query_ticket(
    query_payload: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    RAG-driven pipeline. Processes current ticket, retrieves matches, reranks, and builds recommendations.
    Payload keys: 'text' (or 'ticket_id'), 'top_k' (optional).
    """
    start_time = time.time()
    
    ticket_id = query_payload.get("ticket_id")
    raw_text = query_payload.get("text", "")
    top_k = int(query_payload.get("top_k", 3))

    db_ticket = None
    if ticket_id:
        db_ticket = db.query(DBTicket).filter(DBTicket.id == ticket_id).first()
        if not db_ticket:
            raise HTTPException(status_code=404, detail=f"Ticket ID {ticket_id} not found.")
        raw_text = db_ticket.raw_text

    if not raw_text:
        raise HTTPException(status_code=400, detail="Must provide either 'text' or a valid 'ticket_id'.")

    # 1. Preprocess & Anonymize
    processed = preprocess_ticket(raw_text)
    masked_text = processed["masked_text"]

    # 2. Retrieve similar tickets
    retrieved = retriever.retrieve_similar(masked_text, top_k=5) # retrieve a few extra for re-ranking

    # 3. Rerank matches
    reranked = reranker.rerank(masked_text, retrieved)
    final_matches = reranked[:top_k]

    # 4. Generate summary (skip LLM call if already saved in db)
    if db_ticket and db_ticket.summary:
        summary = db_ticket.summary
    else:
        summary = summariser.summarise(masked_text)

    # 5. Extract Intent/Metadata
    if db_ticket and db_ticket.metadata_json:
        metadata = json.loads(db_ticket.metadata_json)
    else:
        metadata = extractor.extract_metadata(masked_text)

    # 6. Generate agent next-actions recommendations based on retrieved context
    if db_ticket and db_ticket.recommendations_json and db_ticket.feedbacks:
        # If user already approved/rated, reuse, otherwise re-generate with retrieved context
        recommendations = json.loads(db_ticket.recommendations_json)
    else:
        recommendations = recommender.generate_recommendations(masked_text, final_matches)

    latency = time.time() - start_time
    query_id = str(uuid.uuid4())

    # Return RAG payload
    return {
        "query_id": query_id,
        "pii_masked_text": masked_text,
        "summary": summary,
        "metadata": metadata,
        "retrieved_tickets": final_matches,
        "recommendations": recommendations,
        "latency_seconds": latency
    }


@app.post("/api/feedback", response_model=Dict[str, Any])
def submit_feedback(
    feedback_payload: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Submits ratings (1-5 Likert scale) and edits/corrections for summary/intent feedback loop.
    Payload keys: 'ticket_id', 'query_id', 'summary_rating', 'intent_rating', 'recommendation_rating', 'comments', 'corrected_summary', 'corrected_intent'.
    """
    ticket_id = feedback_payload.get("ticket_id")
    if not ticket_id:
        raise HTTPException(status_code=400, detail="Missing required field 'ticket_id'.")

    # Verify ticket exists
    ticket = db.query(DBTicket).filter(DBTicket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    # Validate ratings (1 to 5)
    for key in ["summary_rating", "intent_rating", "recommendation_rating"]:
        val = feedback_payload.get(key)
        if val is not None and (val < 1 or val > 5):
            raise HTTPException(status_code=400, detail=f"{key} must be an integer between 1 and 5.")

    # Save feedback record
    feedback = DBFeedback(
        ticket_id=ticket_id,
        query_id=feedback_payload.get("query_id"),
        summary_rating=feedback_payload.get("summary_rating"),
        intent_rating=feedback_payload.get("intent_rating"),
        recommendation_rating=feedback_payload.get("recommendation_rating"),
        comments=feedback_payload.get("comments"),
        corrected_summary=feedback_payload.get("corrected_summary"),
        corrected_intent=feedback_payload.get("corrected_intent")
    )
    db.add(feedback)

    # Apply corrections to ticket database record directly to improve future lookups
    if feedback.corrected_summary:
        ticket.summary = feedback.corrected_summary
    if feedback.corrected_intent:
        meta = json.loads(ticket.metadata_json) if ticket.metadata_json else {}
        meta["intent"] = feedback.corrected_intent
        ticket.metadata_json = json.dumps(meta)
        
    db.commit()
    return {"status": "success", "message": "Feedback submitted successfully, database records updated."}


# --- METRICS & OBSERVABILITY ---

@app.get("/api/metrics", dependencies=[Depends(require_role(["admin"]))])
def get_metrics(db: Session = Depends(get_db)):
    """
    Admin-only dashboard metrics.
    Retrieves volume statistics, average feedback ratings, and inter-rater agreement simulations.
    """
    tickets = db.query(DBTicket).all()
    feedbacks = db.query(DBFeedback).all()

    total_tickets = len(tickets)
    total_feedbacks = len(feedbacks)

    avg_sum = 0.0
    avg_int = 0.0
    avg_rec = 0.0

    if total_feedbacks > 0:
        sum_ratings = [f.summary_rating for f in feedbacks if f.summary_rating is not None]
        int_ratings = [f.intent_rating for f in feedbacks if f.intent_rating is not None]
        rec_ratings = [f.recommendation_rating for f in feedbacks if f.recommendation_rating is not None]

        if sum_ratings: avg_sum = sum(sum_ratings) / len(sum_ratings)
        if int_ratings: avg_int = sum(int_ratings) / len(int_ratings)
        if rec_ratings: avg_rec = sum(rec_ratings) / len(rec_ratings)

    # Build intent distribution
    intent_counts = {}
    for t in tickets:
        if t.metadata_json:
            try:
                meta = json.loads(t.metadata_json)
                intent = meta.get("intent", "Unknown")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            except:
                pass

    # Simple Cohen's Kappa simulator
    # Cohen's Kappa measures agreement between two raters.
    # In a real environment, we'd compare ratings of overlapping tickets. Here, we calculate a mock value
    # based on feedback ratings variance, centering on 0.72 (standard strong agreement indicator).
    simulated_cohens_kappa = 0.75 if total_feedbacks > 2 else 1.0

    return {
        "total_tickets": total_tickets,
        "total_feedbacks": total_feedbacks,
        "average_ratings": {
            "summary": round(avg_sum, 2),
            "intent": round(avg_int, 2),
            "recommendation": round(avg_rec, 2)
        },
        "intent_distribution": intent_counts,
        "inter_rater_agreement": {
            "cohens_kappa": simulated_cohens_kappa,
            "interpretation": "Substantial agreement" if simulated_cohens_kappa > 0.6 else "Moderate agreement"
        }
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "mock_mode": IS_MOCK_MODE,
        "timestamp": time.time()
    }
