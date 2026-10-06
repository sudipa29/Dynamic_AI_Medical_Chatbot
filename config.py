import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    """Application configuration."""
    
    APP_NAME = "Dynamic AI Chatbot"
    APP_VERSION = "1.0.0"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    MAX_MESSAGE_LENGTH = 1000

    DEFAULT_LANGUAGE = "English"
    DEFAULT_TEMPERATURE = 0.7

    ENABLE_SENTIMENT = True
    ENABLE_CONTEXT = True
    SHOW_TIMESTAMPS = True