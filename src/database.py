import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from src.config import DB_URL

Base = declarative_base()

class DBTicket(Base):
    __tablename__ = 'tickets'

    id = Column(String(50), primary_key=True)
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    masked_text = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)  # Store JSON string
    recommendations_json = Column(Text, nullable=True)  # Store JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    feedbacks = relationship("DBFeedback", back_populates="ticket", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "masked_text": self.masked_text,
            "summary": self.summary,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
            "recommendations": json.loads(self.recommendations_json) if self.recommendations_json else {},
            "created_at": self.created_at.isoformat()
        }


class DBFeedback(Base):
    __tablename__ = 'feedbacks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(50), ForeignKey('tickets.id', ondelete='CASCADE'), nullable=False)
    query_id = Column(String(50), nullable=True)
    summary_rating = Column(Integer, nullable=True)  # 1-5 scale
    intent_rating = Column(Integer, nullable=True)   # 1-5 scale
    recommendation_rating = Column(Integer, nullable=True)  # 1-5 scale
    comments = Column(Text, nullable=True)
    corrected_summary = Column(Text, nullable=True)
    corrected_intent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("DBTicket", back_populates="feedbacks")

    def to_dict(self):
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "query_id": self.query_id,
            "summary_rating": self.summary_rating,
            "intent_rating": self.intent_rating,
            "recommendation_rating": self.recommendation_rating,
            "comments": self.comments,
            "corrected_summary": self.corrected_summary,
            "corrected_intent": self.corrected_intent,
            "created_at": self.created_at.isoformat()
        }


# Database setup
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Initializes/creates tables.
    """
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
