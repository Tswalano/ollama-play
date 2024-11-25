from flask import Blueprint, request, jsonify, render_template
from app.database.manager import DatabaseManager
from app.rag.manager import RAGManager
from app.config import settings
from app.utils.logger import logger
from app.utils.middleware import handle_errors
import markdown
import bleach

api = Blueprint('api', __name__)
db_manager = DatabaseManager(settings.DB_URL)
rag_manager = RAGManager(settings)

class MessageFormatter:
    ALLOWED_TAGS = [
        'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'hr',
        'ul', 'ol', 'li', 'strong', 'em', 'code', 'pre',
        'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'div', 'span', 'a'
    ]
    ALLOWED_ATTRIBUTES = {
        '*': ['class'],
        'a': ['href', 'title', 'target'],
        'code': ['class'],
        'pre': ['class']
    }

    @classmethod
    def format_message(cls, content: str) -> str:
        """Format message content with proper HTML"""
        # Convert markdown to HTML
        html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])
        
        # Clean HTML
        clean_html = bleach.clean(
            html_content,
            tags=cls.ALLOWED_TAGS,
            attributes=cls.ALLOWED_ATTRIBUTES
        )
        
        return clean_html

@api.route('/conversation', methods=['POST'])
@handle_errors
def create_conversation():
    data = request.json
    title = data.get('title') if data else None
    conversation = db_manager.create_conversation(title)
    return jsonify(conversation)

@api.route('/conversation/<int:conversation_id>/chat', methods=['POST'])
@handle_errors
def chat(conversation_id):
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = data['message']
    
    # Save user message
    db_manager.add_message(conversation_id, "user", user_message)
    
    # Get RAG response
    response = rag_manager.query(user_message)
    
    # Format the response
    formatted_response = MessageFormatter.format_message(response)
    
    # Save assistant response
    db_manager.add_message(conversation_id, "assistant", response)
    
    return jsonify({
        "response": response,
        "html": formatted_response
    })

@api.route('/conversation/<int:conversation_id>', methods=['GET'])
@handle_errors
def get_conversation(conversation_id):
    messages = db_manager.get_conversation(conversation_id)
    
    # Format messages for display
    formatted_messages = []
    for msg in messages:
        
        # print(msg)
        formatted_messages.append({
            "role": msg["role"],
            "content": MessageFormatter.format_message(msg["content"]),
            "timestamp": msg["timestamp"]
        })
    
    return jsonify({"conversation": formatted_messages, "conversationId": conversation_id})

@api.route('/conversations', methods=['GET'])
@handle_errors
def get_conversations():
    conversations = db_manager.get_conversations()
    return jsonify({"conversations": conversations})

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
    try:
        db_manager.get_conversations()
        return "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return "unhealthy"

def check_rag_health():
    try:
        rag_manager.query("test query")
        return "healthy"
    except Exception as e:
        logger.error(f"RAG health check failed: {str(e)}")
        return "unhealthy"