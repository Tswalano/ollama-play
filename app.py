from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage, Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import markdown
import bleach
import uuid

app = Flask(__name__)
Base = declarative_base()

# Database Models
class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    uuid = Column(String(36), unique=True)  # Add this line
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    title = Column(String(200))  # For sidebar display
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    conversation = relationship("Conversation", back_populates="messages")

class MessageFormatter:
    @staticmethod
    def format_message(content: str, role: str) -> str:
        """Format message content with Tailwind CSS classes and handle lists"""
        # Handle markdown to HTML conversion
        html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])

        # Clean HTML
        allowed_tags = [
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'hr',
            'ul', 'ol', 'li', 'strong', 'em', 'code', 'pre',
            'table', 'thead', 'tbody', 'tr', 'th', 'td'
        ]
        allowed_attributes = {'*': ['class']}
        clean_html = bleach.clean(html_content, tags=allowed_tags, attributes=allowed_attributes)
        
        # Format response based on the role
        if role == "user":
            return f"""
            <div class="flex justify-end mb-4">
                <div class="bg-primary text-primary-foreground rounded-lg py-2 px-4 max-w-[80%]">
                    {clean_html}
                </div>
            </div>
            """
        else:
            return f"""
            <div class="flex justify-start mb-4">
                <div class="bg-card border border-border rounded-lg py-2 px-4 max-w-[80%]">
                    <div class="prose prose-invert">
                        {clean_html}
                    </div>
                </div>
            </div>
            """

class DatabaseManager:
    def __init__(self, db_url="sqlite:///conversations.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def start_conversation(self, title: str = None, uuid: str = None) -> Dict:
        try:
            session = self.Session()
            conv = Conversation(title=title, uuid=uuid)  # Use the uuid here
            session.add(conv)
            session.commit()
            result = {"id": conv.id, "title": conv.title, "uuid": conv.uuid}
            session.close()
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def add_message(self, conv_id: int, role: str, content: str):
        try:
            session = self.Session()
            message = Message(
                conversation_id=conv_id,
                role=role,
                content=content
            )
            session.add(message)
            session.commit()
            session.close()
        except Exception as e:
            print(f"Error adding message: {e}")
    
    def get_conversation_history(self, conv_id: int) -> List[Dict]:
        try:
            session = self.Session()
            messages = session.query(Message).filter(
                Message.conversation_id == conv_id
            ).order_by(Message.timestamp).all()
            
            history = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
            session.close()
            return history
        except Exception as e:
            return {"error": str(e)}
    
    def get_all_conversations(self) -> List[Dict]:
        try:
            session = self.Session()
            conversations = session.query(Conversation).order_by(
                Conversation.start_time.desc()
            ).all()
            
            result = [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "start_time": conv.start_time.isoformat()
                }
                for conv in conversations
            ]
            session.close()
            return result
        except Exception as e:
            return {"error": str(e)}

# Database Manager Instance
db_manager = DatabaseManager()


# Load financials, departments, and users data
financials_df = pd.read_csv("./data/financials.csv")
departments_df = pd.read_csv("./data/departments.csv")
users_df = pd.read_csv("./data/users.csv")


# Convert DataFrames into lists of dictionaries
financials = financials_df.to_dict(orient="records")
departments = departments_df.to_dict(orient="records")
users = users_df.to_dict(orient="records")


# Combine all documents into a single list
documents = financials + departments + users


# Define the LLaMa API call (or another language model)
def call_llama(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        headers=headers,
        json=payload
    )
    return response.json()["response"]

# Create a custom LLaMa class
class LLaMa(LLM):
    def _call(self, prompt, **kwargs):
        return call_llama(prompt)

    @property
    def _llm_type(self):
        return "llama-3.1-8b"


# Extract document texts for FAISS retrieval
texts = [str(doc) for doc in documents]  
retriever = FAISS.from_texts(
    texts,
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
).as_retriever(k=5)


# Define the FAQ prompt template
faq_template = """
You are a chat agent for my business. Help answer questions based on the following data:

{context}
"""

faq_prompt = ChatPromptTemplate.from_messages([
    ("system", faq_template),
    MessagesPlaceholder("messages")
])


# Create a document and retrieval chain
document_chain = LLMChain(llm=LLaMa(), prompt=faq_prompt)


# Function to parse the retriever input
def parse_retriever_input(params):
    return params["messages"][-1].content


# Define the retrieval chain
retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever
).assign(answer=document_chain)


# RAG Chain
rag_chain = retrieval_chain


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/conversation/<int:conversationId>/chat", methods=["POST"])
def chat(conversationId):
    data = request.json
    user_query = data.get("query")
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    response = rag_chain.invoke({
        "messages": [HumanMessage(user_query)]
    })
    
    # Extract the plain text answer (without HTML formatting)
    answer_text = response["answer"]["text"]
    
    # Add message to the database
    db_manager.add_message(conversationId, "assistant", answer_text)
    db_manager.add_message(conversationId, "user", user_query)
    
    # Return the plain text answer and conversation ID
    return jsonify({
        "text": answer_text,
        "conversationId": conversationId
    })



@app.route("/conversation/<int:conversationId>", methods=["GET"])
def get_conversation(conversationId):
    # Fetch the conversation history from the database
    conversation = db_manager.get_conversation_history(conversationId)
    
    return jsonify({"conversation": conversation})


@app.route("/conversations", methods=["GET"])
def get_conversations():
    conversations = db_manager.get_all_conversations()
    return jsonify({"conversations": conversations})

@app.route("/conversation", methods=["POST"])
def create_conversation():
    
    # generate a uuid
    conversation_uuid = str(uuid.uuid4())
    
    # string Conversation title with date time like Conversation 24 Sep 24 @ 11:12
    now = datetime.now()
    date_time = now.strftime("%d %b %Y @ %H:%M")
    title = "Conversation " + date_time
    
    data = request.json
    title = title or data.get("title", title)
    
    conversation = db_manager.start_conversation(title, conversation_uuid)
    
    return jsonify(conversation)


if __name__ == "__main__":
    app.run(debug=True)
