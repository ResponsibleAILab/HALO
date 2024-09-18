import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configurations
OPENAI_API_TYPE = ""
OPENAI_API_BASE = os.getenv('')
OPENAI_API_KEY = os.getenv("")
OPENAI_API_VERSION = os.getenv('')

# File paths
QUESTIONS_FILE = ""
DOCUMENTS_PATH = ""
