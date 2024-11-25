from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    uuid = Column(String(36), unique=True, nullable=False, default=generate_uuid)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime)
    title = Column(String(200), nullable=False)
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    message_uuid = Column(String(36), unique=True, nullable=False, default=generate_uuid)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    conversation = relationship("Conversation", back_populates="messages")