from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI()

def test_web_search_function():
    logger.info("Testing web search function tool...")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }]
        )
        logger.info("Success! OpenAI API accepted the tools parameter structure.")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_web_search_function()
