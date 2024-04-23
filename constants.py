import os
from dotenv import load_dotenv


load_dotenv()
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
WAND_API_KEY = os.getenv('WAND_API_KEY')
