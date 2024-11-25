from flask import Flask, request, jsonify, render_template
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import uuid
import logging
from pathlib import Path
from typing import List, Dict
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from pydantic_settings import BaseSettings
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Settings
class Settings(BaseSettings):
    DATA_DIR: Path = Path("./data")
    CHROMA_DIR: Path = Path("./chroma_db")
    DB_URL: str = "sqlite:///conversations.db"
    LLM_MODEL: str = "llama3.2:1b"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    
    SYSTEM_PROMPT: str = """
    You are a helpful assistant that provides accurate responses based on the given context.
    Answer questions about the company's data including employees, departments, and financials.
    
    Context Information:
    {context}
    
    Question: {question}
    
    Provide a clear and accurate response based on the context."""

# Database Models
Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    # uuid = Column(String(36), unique=True, nullable=False)
    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
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
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    conversation = relationship("Conversation", back_populates="messages")

class DatabaseManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        # Drop all tables and recreate them
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_conversation(self, title: str = None) -> Dict:
        try:
            session = self.Session()
            conversation_uuid = str(uuid.uuid4())
            if not title:
                title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            conv = Conversation(uuid=conversation_uuid, title=title)
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
                uuid=str(uuid.uuid4()),  # Explicitly set UUID
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

class RAGManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        self.llm = Ollama(model=settings.LLM_MODEL)
        self.vectorstore = None
        self.qa_chain = None
        self.initialize_rag()
    
    def initialize_rag(self):
        try:
            # Load your CSV data
            employees = pd.read_csv(self.settings.DATA_DIR / "employees.csv")
            departments = pd.read_csv(self.settings.DATA_DIR / "departments.csv")
            financials = pd.read_csv(self.settings.DATA_DIR / "financials.csv")
            
            # Create documents for vector store
            documents = []
            
            # Process employees with departments
            emp_dept = pd.merge(
                employees,
                departments,
                left_on='department_id',
                right_on='id',
                suffixes=('_emp', '_dept')
            )
            
            for _, row in emp_dept.iterrows():
                doc = f"""
                Employee: {row['first_name']} {row['last_name']}
                Position: {row['position']}
                Department: {row['name']}
                Salary: ${row['salary']}
                Hire Date: {row['hire_date']}
                """
                documents.append(doc)
            
            # Process financials
            fin_dept = pd.merge(
                financials,
                departments,
                left_on='department_id',
                right_on='id'
            )
            
            for _, row in fin_dept.iterrows():
                doc = f"""
                Department: {row['name']}
                Year: {row['year']}
                Quarter: Q{row['quarter']}
                Revenue: ${row['revenue']}
                Expenses: ${row['expenses']}
                Profit: ${row['profit']}
                """
                documents.append(doc)
            
            # Create vector store
            self.vectorstore = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                persist_directory=str(self.settings.CHROMA_DIR)
            )
            
            # Create QA chain
            prompt = PromptTemplate(
                template=self.settings.SYSTEM_PROMPT,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": prompt}
            )
            
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            raise
    
    def query(self, question: str) -> str:
        try:
            response = self.qa_chain.invoke({"query": question})
            return response['result']
        except Exception as e:
            logger.error(f"Error querying RAG: {str(e)}")
            raise

# Flask Application
app = Flask(__name__)

# Initialize components
settings = Settings()
db_manager = DatabaseManager(settings.DB_URL)
rag_manager = RAGManager(settings)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/conversation', methods=['POST'])
def create_conversation():
    try:
        data = request.json
        title = data.get('title') if data else None
        conversation = db_manager.create_conversation(title)
        return jsonify(conversation)
    except Exception as e:
        logger.error(f"Error in create_conversation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation/<int:conversation_id>/messages', methods=['POST'])
def add_message(conversation_id):
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message']
        
        # Save user message
        db_manager.add_message(conversation_id, "user", user_message)
        
        # Get RAG response
        response = rag_manager.query(user_message)
        
        # Save assistant response
        db_manager.add_message(conversation_id, "assistant", response)
        
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in add_message: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    try:
        messages = db_manager.get_conversation(conversation_id)
        return jsonify({"messages": messages})
    except Exception as e:
        logger.error(f"Error in get_conversation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    try:
        conversations = db_manager.get_conversations()
        return jsonify({"conversations": conversations})
    except Exception as e:
        logger.error(f"Error in get_conversations: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)