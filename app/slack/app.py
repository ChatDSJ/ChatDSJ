from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
import random
from datetime import datetime
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    app = App(token=os.environ.get("SLACK_BOT_TOKEN"),
              signing_secret=os.environ.get("SLACK_SIGNING_SECRET"))
    logger.info("Slack app initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Slack app: {e}")
    class DummyApp:
        def event(self, event_type):
            def decorator(func):
                return func
            return decorator
        def error(self, func):
            return func
    app = DummyApp()

openai_client = None
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")

channel_data = {}

SYSTEM_PROMPT = """
You are a helpful assistant participating in a Slack conversation.
Your job is to provide helpful, informative, and concise responses based on the conversation context.
You will be given the entire conversation history of the channel, followed by the message in which you were mentioned.
Respond in a way that is helpful and relevant to the conversation, considering all the previous context.
Keep your responses conversational and friendly.
"""

def update_channel_stats(channel_id, user_id, message_ts):
    """Update the channel statistics"""
    if channel_id not in channel_data:
        channel_data[channel_id] = {
            "message_count": 0,
            "participants": set(),
            "last_updated": datetime.now()
        }

    channel_data[channel_id]["message_count"] += 1
    channel_data[channel_id]["participants"].add(user_id)
    channel_data[channel_id]["last_updated"] = datetime.now()

def get_channel_stats(channel_id):
    """Get the channel statistics"""
    if channel_id not in channel_data:
        return {
            "message_count": 0,
            "participants": set(),
            "last_updated": datetime.now()
        }

    return channel_data[channel_id]

def get_channel_history(client, channel_id):
    """Get the conversation history of the channel"""
    try:
        result = client.conversations_history(channel=channel_id, limit=100)
        return result["messages"]
    except Exception as e:
        logger.error(f"Error fetching channel history: {e}")
        return []

def format_conversation_history(messages, client):
    """Format the conversation history for the ChatGPT prompt"""
    formatted_messages = []

    for message in reversed(messages):
        user_id = message.get("user", "unknown")
        text = message.get("text", "")

        try:
            user_info = client.users_info(user=user_id)
            username = user_info["user"]["real_name"]
        except Exception:
            username = f"User {user_id}"

        formatted_messages.append(f"{username}: {text}")

    return "\n".join(formatted_messages)

def get_chatgpt_response(conversation_history, current_message):
    """Get a response from ChatGPT based on the conversation history and current message"""
    if not openai_client:
        return "I'm having trouble connecting to my brain right now. Please try again later."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation history:\n{conversation_history}\n\nCurrent message: {current_message}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting ChatGPT response: {e}")
        return "I'm having trouble thinking right now. Please try again later."

@app.event("app_mention")
def handle_mention(event, say, client):
    channel_id = event["channel"]
    user_id = event["user"]
    message_ts = event["ts"]
    message_text = event["text"]

    update_channel_stats(channel_id, user_id, message_ts)
    stats = get_channel_stats(channel_id)

    channel_history = get_channel_history(client, channel_id)
    conversation_context = format_conversation_history(channel_history, client)

    chatgpt_response = get_chatgpt_response(conversation_context, message_text)

    participants_list = ", ".join([f"<@{user}>" for user in stats["participants"]])
    channel_stats_text = f"This channel has {stats['message_count']} messages from {len(stats['participants'])} participants: {participants_list}"

    say(f"{chatgpt_response}")

@app.error
def error_handler(error, body, logger):
    logger.error(f"Error: {error}")
    logger.debug(f"Body: {body}")
