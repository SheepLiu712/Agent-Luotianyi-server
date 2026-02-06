from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import uuid
import os
import json

Base = declarative_base()

class Song(Base):
    __tablename__ = "songs"
    
    uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    safe_name = Column(String, nullable=False)
    uploader = Column(String, nullable=True) # UP主
    singers = Column(String, nullable=True) # 演唱
    introduction = Column(Text, nullable=False) # summary
    lyrics = Column(Text, nullable=False) # lyrics (cleaned)

SessionLocal = None
engine = None

def init_song_db(db_folder: str = None, db_file: str = None):
    """Initialize database tables"""
    global engine, SessionLocal
    
    # Ensure directory exists
    if not os.path.exists(db_folder):
        os.makedirs(db_folder, exist_ok=True)
        
    DATABASE_URL = f"sqlite:///{os.path.join(db_folder, db_file)}"

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

def get_song_db():
    """Generator for database session"""
    if SessionLocal is None:
         # Fallback default path if not initialized explicitly
        init_song_db(
            db_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "res", "knowledge"),
            db_file="knowledge_db.db"
        )
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_song_session():
    """Direct session"""
    if SessionLocal is None:
         # Fallback default path if not initialized explicitly
        init_song_db(
            db_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "res", "knowledge"),
            db_file="knowledge_db.db"
        )
    return SessionLocal()


