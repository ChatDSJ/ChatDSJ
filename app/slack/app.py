from slack_bolt import App
import os
import random
from datetime import datetime

app = App(token=os.environ.get("SLACK_BOT_TOKEN"),
          signing_secret=os.environ.get("SLACK_SIGNING_SECRET"))

channel_data = {}

RUDE_PHRASES = [
    "Do I look like I care?",
    "Not this again...",
    "Are you seriously bothering me right now?",
    "I have better things to do than talk to you.",
    "Oh great, another pointless conversation.",
    "Whatever. I'm busy.",
    "That's the dumbest thing I've heard all day.",
    "Can you not?",
    "Ugh, what now?",
    "You again? Seriously?"
]

def get_random_rude_phrase():
    """Return a randomly selected rude phrase"""
    return random.choice(RUDE_PHRASES)

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

@app.event("app_mention")
def handle_mention(event, say):
    channel_id = event["channel"]
    user_id = event["user"]
    message_ts = event["ts"]
    
    update_channel_stats(channel_id, user_id, message_ts)
    stats = get_channel_stats(channel_id)
    
    
    response = get_random_rude_phrase()
    
    participants_list = ", ".join([f"<@{user}>" for user in stats["participants"]])
    channel_stats_text = f"This channel has {stats['message_count']} messages from {len(stats['participants'])} participants: {participants_list}"
    
    say(f"{response}\n\n{channel_stats_text}")

@app.error
def error_handler(error, body, logger):
    logger.error(f"Error: {error}")
    logger.debug(f"Body: {body}")
