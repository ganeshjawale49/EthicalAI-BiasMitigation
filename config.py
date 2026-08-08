# System Configuration Settings for Ethical AI Bias Mitigation Application
import os
import tempfile

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IS_VERCEL = bool(os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ethical_ai_bias_mitigation_secret_key_2026')
    BASE_DIR = BASE_DIR
    
    # In Vercel or read-only environments, use /tmp for SQLite database and uploaded files
    if IS_VERCEL or not os.access(BASE_DIR, os.W_OK):
        TEMP_DIR = tempfile.gettempdir()
        DATABASE_PATH = os.path.join(TEMP_DIR, 'ethical_ai_database.db')
        UPLOAD_FOLDER = os.path.join(TEMP_DIR, 'ethical_ai_uploads')
    else:
        DATABASE_PATH = os.path.join(BASE_DIR, 'database.db')
        UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload limit
    ALLOWED_EXTENSIONS = {'csv'}
    
    # LLM API Configurations
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

