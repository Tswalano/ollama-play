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
    app.run(debug=True, host='0.0.0.0', port=5001)