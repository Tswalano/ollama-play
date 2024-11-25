from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict
from datetime import datetime
from .models import Base, Conversation, Message
from app.utils.logger import logger

class DatabaseManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_conversation(self, title: str = None) -> Dict:
        try:
            session = self.Session()
            if not title:
                title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            conv = Conversation(title=title)
            session.add(conv)
            session.commit()
            
            result = {
                "id": conv.id,
                "uuid": conv.uuid,
                "title": conv.title
            }
            session.close()
            return result
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise
    
    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        try:
            session = self.Session()
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
            session.add(message)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise
    
    def get_conversation(self, conversation_id: int) -> List[Dict]:
        try:
            session = self.Session()
            messages = session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.timestamp).all()
            
            result = [
                {
                    "uuid": msg.message_uuid,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
            session.close()
            return result
        except Exception as e:
            logger.error(f"Error retrieving conversation: {str(e)}")
            raise
    
    def get_conversations(self) -> List[Dict]:
        try:
            session = self.Session()
            conversations = session.query(Conversation).order_by(
                Conversation.start_time.desc()
            ).all()
            
            result = [
                {
                    "id": conv.id,
                    "uuid": conv.uuid,
                    "title": conv.title,
                    "start_time": conv.start_time.isoformat()
                }
                for conv in conversations
            ]
            session.close()
            return result
        except Exception as e:
            logger.error(f"Error retrieving conversations: {str(e)}")
            raise