import pytest
from src.ingestion import clean_text, parse_speaker_segments, mask_pii, preprocess_ticket
from src.embeddings import MockEmbeddingModel
from src.reranker import TicketReranker

def test_text_cleaning():
    raw = "   Some    support message   with \n newlines.   "
    cleaned = clean_text(raw)
    assert cleaned == "Some support message with newlines."

def test_speaker_splitting():
    transcript = (
        "Customer: Hello, I have an issue.\n"
        "Agent: Welcome! I can help you.\n"
        "Customer: Thanks, it's about my refund."
    )
    segments = parse_speaker_segments(transcript)
    assert len(segments) == 3
    assert segments[0]["speaker"] == "Customer"
    assert segments[0]["text"] == "Hello, I have an issue."
    assert segments[1]["speaker"] == "Agent"
    assert segments[1]["text"] == "Welcome! I can help you."

def test_pii_masking():
    sensitive_text = "Contact me at bob.smith@company.com or call 555-123-4567. SSN is 123-45-6789. Card: 4111-2222-3333-4444."
    masked = mask_pii(sensitive_text)
    assert "[EMAIL]" in masked
    assert "[PHONE]" in masked
    assert "[SSN]" in masked
    assert "[CREDIT_CARD]" in masked
    assert "bob.smith" not in masked
    assert "555-123" not in masked

def test_preprocessing_pipeline():
    raw = "Customer: hi my email is test@domain.com"
    processed = preprocess_ticket(raw)
    assert "segments" in processed
    assert len(processed["segments"]) == 1
    assert processed["segments"][0]["text"] == "hi my email is [EMAIL]"

def test_mock_embeddings():
    embedder = MockEmbeddingModel(dimension=128)
    v1 = embedder.embed_query("billing issue")
    v2 = embedder.embed_query("billing issue")
    v3 = embedder.embed_query("something else")
    
    assert len(v1) == 128
    # Test determinism
    assert v1 == v2
    # Test variance
    assert v1 != v3

def test_reranker_fallback():
    query = "double subscription charge refund"
    tickets = [
        {"text": "I was double charged on my visa card, please refund me"},
        {"text": "How do I reset my password and login?"},
        {"text": "The web interface is running slowly today"}
    ]
    
    reranker = TicketReranker(use_transformer=False)
    sorted_results = reranker.rerank(query, tickets)
    
    assert len(sorted_results) == 3
    # The double charge ticket should be ranked first due to high keyword overlap/TF-IDF similarity
    assert "double charged" in sorted_results[0]["text"]
    assert sorted_results[0]["rerank_score"] > sorted_results[1]["rerank_score"]
