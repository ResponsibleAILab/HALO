import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configurations
OPENAI_API_TYPE = "azure"
OPENAI_API_BASE = os.getenv('https://sumera.openai.azure.com/')
OPENAI_API_KEY = os.getenv("7a271e053b8340bebe993132c4cb1b05")
OPENAI_API_VERSION = os.getenv('2023-07-01-preview')

# File paths
QUESTIONS_FILE = "c:/Users/asume/Downloads/SummuProject/SummuProject/train2.json"
DOCUMENTS_PATH = "c:/Users/asume/Downloads/SummuProject/SummuProject/"
