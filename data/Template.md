1. First, `app/config.py`:
```python
from pydantic_settings import BaseSettings
from pathlib import Path

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

    class Config:
        env_file = ".env"

settings = Settings()
```

2. `app/utils/logger.py`:
```python
import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logger()
```

3. `app/database/models.py`:
```python
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
```

4. `app/database/manager.py`:
```python
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
```

5. `app/rag/manager.py`:
```python
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from app.utils.logger import logger
from app.config import Settings

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
            # Load CSV data
            employees = pd.read_csv(self.settings.DATA_DIR / "employees.csv")
            departments = pd.read_csv(self.settings.DATA_DIR / "departments.csv")
            financials = pd.read_csv(self.settings.DATA_DIR / "financials.csv")
            
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
            
            logger.info("RAG system initialized successfully")
            
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
```

6. `app/api/routes.py`:
```python
from flask import Blueprint, request, jsonify
from app.database.manager import DatabaseManager
from app.rag.manager import RAGManager
from app.config import settings
from app.utils.logger import logger

api = Blueprint('api', __name__)
db_manager = DatabaseManager(settings.DB_URL)
rag_manager = RAGManager(settings)

@api.route('/conversation', methods=['POST'])
def create_conversation():
    try:
        data = request.json
        title = data.get('title') if data else None
        conversation = db_manager.create_conversation(title)
        return jsonify(conversation)
    except Exception as e:
        logger.error(f"Error in create_conversation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api.route('/conversation/<int:conversation_id>/messages', methods=['POST'])
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
        
        return jsonify({
            "response": response
        })
    except Exception as e:
        logger.error(f"Error in add_message: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api.route('/conversation/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    try:
        messages = db_manager.get_conversation(conversation_id)
        return jsonify({"messages": messages})
    except Exception as e:
        logger.error(f"Error in get_conversation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api.route('/conversations', methods=['GET'])
def get_conversations():
    try:
        conversations = db_manager.get_conversations()
        return jsonify({"conversations": conversations})
    except Exception as e:
        logger.error(f"Error in get_conversations: {str(e)}")
        return jsonify({"error": str(e)}), 500
```

7. `main.py`:
```python
from flask import Flask, render_template
from app.api.routes import api
from app.config import settings
from app.utils.logger import logger
from pathlib import Path

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    return app

def ensure_directories():
    """Ensure required directories exist"""
    settings.DATA_DIR.mkdir(exist_ok=True)
    settings.CHROMA_DIR.mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    app = create_app()
    app.run(debug=True)
```

8. Sample data files:

`data/employees.csv`:
```csv
id,first_name,last_name,department_id,position,salary,hire_date,manager_id
1,John,Smith,1,Lead Engineer,120000,2020-01-15,
2,Jane,Doe,2,Sales Director,150000,2019-03-20,
3,Bob,Johnson,3,Marketing Head,140000,2019-06-10,
4,Alice,Williams,4,HR Director,130000,2019-09-05,
5,Charlie,Brown,1,Senior Engineer,100000,2020-04-15,1
```

`data/departments.csv`:
```csv
id,name,head_id,budget,location
1,Engineering,1,1000000,Floor 3
2,Sales,2,800000,Floor 2
3,Marketing,3,600000,Floor 2
4,HR,4,400000,Floor 1
```

`data/financials.csv`:
```csv
id,department_id,year,quarter,revenue,expenses,profit
1,1,2023,1,300000,250000,50000
2,1,2023,2,350000,280000,70000
3,2,2023,1,400000,300000,100000
4,2,2023,2,450000,320000,130000
```

To run the application:

1. Install dependencies:
```bash
pip install flask sqlalchemy langchain langchain-community chromadb pandas pydantic-settings python-dotenv
```

2. Create the directory structure and files as shown above.

3. Make sure Ollama is running with the required models:
```bash
ollama pull llama3.2:1b
ollama pull mxbai-embed-large
```

4. Run the application:
```bash
python main.py
```

Test with Postman using the endpoints:
- POST `/api/conversation`
- POST `/api/conversation/{id}/messages`
- GET `/api/conversation/{id}`
- GET `/api/conversations`

The application will:
- Create and manage conversations
- Process questions using the RAG system
- Store all interactions in the database
- Provide a clean API interface for client interaction

9. `templates/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Business Chat Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .chat-container {
            height: calc(100vh - 200px);
            overflow-y: auto;
        }
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <div class="grid grid-cols-4 gap-4">
            <!-- Sidebar -->
            <div class="col-span-1">
                <div class="bg-white rounded-lg shadow-lg p-4">
                    <h2 class="text-xl font-bold mb-4">Conversations</h2>
                    <div id="conversationsList" class="space-y-2">
                        <!-- Conversations will be listed here -->
                    </div>
                    <button 
                        onclick="createConversation()"
                        class="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                    >
                        New Conversation
                    </button>
                </div>
            </div>
            
            <!-- Main Chat Area -->
            <div class="col-span-3">
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <div class="chat-container mb-6" id="chatMessages">
                        <!-- Messages will be inserted here -->
                    </div>
                    
                    <div class="flex space-x-4">
                        <input 
                            type="text" 
                            id="userInput"
                            class="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="Type your message..."
                        >
                        <button 
                            onclick="sendMessage()"
                            class="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            id="sendButton"
                        >
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentConversationId = null;

        // Load conversations on page load
        window.addEventListener('load', loadConversations);

        async function loadConversations() {
            try {
                const response = await fetch('/api/conversations');
                const data = await response.json();
                const conversationsList = document.getElementById('conversationsList');
                conversationsList.innerHTML = '';
                
                data.conversations.forEach(conv => {
                    const button = document.createElement('button');
                    button.className = `w-full text-left px-4 py-2 rounded hover:bg-gray-100 
                        ${conv.id === currentConversationId ? 'bg-blue-100' : ''}`;
                    button.textContent = conv.title;
                    button.onclick = () => loadConversation(conv.id);
                    conversationsList.appendChild(button);
                });
            } catch (error) {
                console.error('Error loading conversations:', error);
                showError('Failed to load conversations');
            }
        }

        async function createConversation() {
            try {
                const response = await fetch('/api/conversation', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                const data = await response.json();
                currentConversationId = data.id;
                await loadConversations();
                document.getElementById('chatMessages').innerHTML = '';
            } catch (error) {
                console.error('Error creating conversation:', error);
                showError('Failed to create conversation');
            }
        }

        async function loadConversation(conversationId) {
            try {
                currentConversationId = conversationId;
                const response = await fetch(`/api/conversation/${conversationId}`);
                const data = await response.json();
                const chatMessages = document.getElementById('chatMessages');
                chatMessages.innerHTML = '';
                
                data.messages.forEach(msg => {
                    addMessage(msg.content, msg.role);
                });
                
                // Update conversation list to show active conversation
                loadConversations();
            } catch (error) {
                console.error('Error loading conversation:', error);
                showError('Failed to load conversation');
            }
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const sendButton = document.getElementById('sendButton');
            const message = input.value.trim();
            
            if (!message) return;
            
            if (!currentConversationId) {
                await createConversation();
            }

            // Disable input and button while processing
            input.disabled = true;
            sendButton.disabled = true;
            
            try {
                addMessage(message, 'user');
                input.value = '';

                const response = await fetch(`/api/conversation/${currentConversationId}/messages`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                if (data.response) {
                    addMessage(data.response, 'assistant');
                }
            } catch (error) {
                console.error('Error sending message:', error);
                showError('Failed to send message');
            } finally {
                // Re-enable input and button
                input.disabled = false;
                sendButton.disabled = false;
                input.focus();
            }
        }

        function addMessage(content, role) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = role === 'user' ? 'flex justify-end mb-4' : 'flex justify-start mb-4';
            
            const innerDiv = document.createElement('div');
            innerDiv.className = role === 'user' 
                ? 'bg-blue-500 text-white rounded-lg py-2 px-4 max-w-[80%]'
                : 'bg-gray-200 rounded-lg py-2 px-4 max-w-[80%]';
            
            const formattedContent = formatMessage(content);
            innerDiv.innerHTML = formattedContent;
            messageDiv.appendChild(innerDiv);
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function formatMessage(content) {
            // Convert line breaks to <br> tags
            return content.replace(/\n/g, '<br>');
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg';
            errorDiv.textContent = message;
            document.body.appendChild(errorDiv);
            
            setTimeout(() => {
                errorDiv.remove();
            }, 3000);
        }

        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
```

10. Add utility functions in `app/utils/data_processor.py`:
```python
import pandas as pd
from typing import List, Dict
from pathlib import Path
from app.utils.logger import logger

def create_sample_data(data_dir: Path):
    """Create sample CSV files if they don't exist"""
    
    # Sample data
    employees_data = {
        'id': range(1, 11),
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 
                     'David', 'Eve', 'Frank', 'Grace', 'Henry'],
        'last_name': ['Smith', 'Doe', 'Johnson', 'Williams', 'Brown',
                     'Davis', 'Wilson', 'Moore', 'Taylor', 'Anderson'],
        'department_id': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
        'position': ['Lead Engineer', 'Sales Director', 'Marketing Head', 'HR Director',
                    'Senior Engineer', 'Sales Manager', 'Marketing Manager', 'HR Manager',
                    'Engineer', 'Sales Representative'],
        'salary': [120000, 150000, 140000, 130000, 100000,
                  90000, 95000, 85000, 80000, 70000],
        'hire_date': ['2020-01-15', '2019-03-20', '2019-06-10', '2019-09-05',
                     '2020-04-15', '2020-07-20', '2020-10-10', '2021-01-05',
                     '2021-04-15', '2021-07-20'],
        'manager_id': [None, None, None, None, 1, 2, 3, 4, 1, 2]
    }
    
    departments_data = {
        'id': range(1, 5),
        'name': ['Engineering', 'Sales', 'Marketing', 'HR'],
        'head_id': [1, 2, 3, 4],
        'budget': [1000000, 800000, 600000, 400000],
        'location': ['Floor 3', 'Floor 2', 'Floor 2', 'Floor 1']
    }
    
    financials_data = {
        'id': range(1, 13),
        'department_id': [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4],
        'year': [2023] * 12,
        'quarter': [1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3],
        'revenue': [300000, 350000, 400000, 450000, 200000, 250000, 100000, 120000,
                   380000, 480000, 280000, 140000],
        'expenses': [250000, 280000, 300000, 320000, 150000, 180000, 80000, 90000,
                    300000, 350000, 200000, 100000],
        'profit': [50000, 70000, 100000, 130000, 50000, 70000, 20000, 30000,
                  80000, 130000, 80000, 40000]
    }
    
    # Create DataFrames
    employees_df = pd.DataFrame(employees_data)
    departments_df = pd.DataFrame(departments_data)
    financials_df = pd.DataFrame(financials_data)
    
    # Save to CSV
    data_dir.mkdir(exist_ok=True)
    employees_df.to_csv(data_dir / 'employees.csv', index=False)
    departments_df.to_csv(data_dir / 'departments.csv', index=False)
    financials_df.to_csv(data_dir / 'financials.csv', index=False)
    
    logger.info("Sample data created successfully")

def process_data_for_rag(data_dir: Path) -> List[str]:
    """Process CSV data into documents for RAG"""
    try:
        employees = pd.read_csv(data_dir / "employees.csv")
        departments = pd.read_csv(data_dir / "departments.csv")
        financials = pd.read_csv(data_dir / "financials.csv")
        
        documents = []
        
        # Process employees with departments
        emp_dept = pd.merge(
            employees,
            departments,
            left_on='department_id',
            right_on='id',
            suffixes=('_emp', '_dept')
        )
        
        # Group by department
        for dept_name, group in emp_dept.groupby('name'):
            emp_list = []
            for _, emp in group.iterrows():
                emp_list.append(f"""
                    {emp['first_name']} {emp['last_name']}:
                    - Position: {emp['position']}
                    - Salary: ${emp['salary']:,}
                    - Hire Date: {emp['hire_date']}
                    """)
            
            doc = f"""
            Department: {dept_name}
            Location: {group.iloc[0]['location']}
            Employees:
            {''.join(emp_list)}
            """
            documents.append(doc)
        
        # Process financials by department and quarter
        fin_dept = pd.merge(
            financials,
            departments,
            left_on='department_id',
            right_on='id'
        )
        
        for dept_name, group in fin_dept.groupby('name'):
            for year in group['year'].unique():
                year_data = group[group['year'] == year]
                quarters_info = []
                for _, quarter in year_data.iterrows():
                    quarters_info.append(f"""
                    Q{quarter['quarter']}:
                    - Revenue: ${quarter['revenue']:,}
                    - Expenses: ${quarter['expenses']:,}
                    - Profit: ${quarter['profit']:,}
                    """)
                
                doc = f"""
                Department: {dept_name}
                Year: {year}
                Financial Results:
                {''.join(quarters_info)}
                """
                documents.append(doc)
        
        return documents
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise
```

11. Add error handling middleware in `app/utils/middleware.py`:
```python
from functools import wraps
from flask import jsonify
from app.utils.logger import logger

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return decorated_function
```

12. Completing `main.py`:
```python
from flask import Flask, render_template
from app.api.routes import api
from app.config import settings
from app.utils.logger import logger
from app.utils.data_processor import create_sample_data
from pathlib import Path
import os

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    return app

def initialize_application():
    """Initialize application with required directories and data"""
    try:
        # Create required directories
        settings.DATA_DIR.mkdir(exist_ok=True)
        settings.CHROMA_DIR.mkdir(exist_ok=True)
        Path('templates').mkdir(exist_ok=True)
        
        # Check if data files exist, create if they don't
        if not all((settings.DATA_DIR / f"{table}.csv").exists() 
                  for table in ['employees', 'departments', 'financials']):
            logger.info("Creating sample data files...")
            create_sample_data(settings.DATA_DIR)
        
        # Initialize ChromaDB directory
        if not os.path.exists(settings.CHROMA_DIR):
            logger.info("Initializing ChromaDB directory...")
            settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize application
    initialize_application()
    
    # Create and run Flask app
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
```

13. Add a health check endpoint in `app/api/routes.py`:
```python
@api.route('/health', methods=['GET'])
@handle_errors
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "components": {
            "database": check_database_health(),
            "rag": check_rag_health()
        }
    })

def check_database_health():
    """Check database connection"""
    try:
        db_manager.get_conversations()
        return "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return "unhealthy"

def check_rag_health():
    """Check RAG system"""
    try:
        # Simple test query
        rag_manager.query("test query")
        return "healthy"
    except Exception as e:
        logger.error(f"RAG health check failed: {str(e)}")
        return "unhealthy"
```

14. Add environment variables support with `.env`:
```plaintext
DB_URL=sqlite:///conversations.db
LLM_MODEL=llama3.2:1b
EMBEDDING_MODEL=mxbai-embed-large
FLASK_ENV=development
```

15. Create a Docker setup:

`Dockerfile`:
```dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data chroma_db templates

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]
```

`docker-compose.yml`:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
    environment:
      - FLASK_ENV=development
      - DB_URL=sqlite:///conversations.db
      - LLM_MODEL=llama3.2:1b
      - EMBEDDING_MODEL=mxbai-embed-large
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

16. Add `requirements.txt`:
```txt
flask>=2.3.3
sqlalchemy>=2.0.23
pydantic-settings>=2.1.0
pandas>=2.1.3
numpy>=1.26.2
requests>=2.31.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-core>=0.1.1
chromadb>=0.4.18
python-dotenv>=1.0.0
```

17. Add `.gitignore`:
```plaintext
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
.env

# Database
*.db
*.sqlite3

# ChromaDB
chroma_db/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
*.log

# OS
.DS_Store
Thumbs.db
```

To run the application:

1. Without Docker:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

2. With Docker:
```bash
# Build and run
docker-compose up --build
```

Test the application:

1. Health Check:
```bash
curl http://localhost:5000/api/health
```

2. Create Conversation:
```bash
curl -X POST http://localhost:5000/api/conversation \
    -H "Content-Type: application/json" \
    -d '{"title": "Test Conversation"}'
```

3. Send Message:
```bash
curl -X POST http://localhost:5000/api/conversation/1/messages \
    -H "Content-Type: application/json" \
    -d '{"message": "How many employees are in the Engineering department?"}'
```

4. Get Conversation History:
```bash
curl http://localhost:5000/api/conversation/1
```

The application now includes:
- Complete error handling
- Docker support
- Environment variable configuration
- Health checks
- Structured logging
- Sample data generation
- Database management
- RAG system with ChromaDB
- Web interface with real-time updates
- API documentation
- Development and deployment configurations

You can access the web interface at http://localhost:5000 and the API endpoints at http://localhost:5000/api/*.