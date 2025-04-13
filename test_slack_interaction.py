import os
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

slack_token = os.environ.get("SLACK_BOT_TOKEN")
client = WebClient(token=slack_token)

response = client.chat_postMessage(
    channel="#chatdsj_testing_ground",
    text="<@U08N3EFH6SE> what day is it?"
)

print(f"Message sent: {response['ts']}")

response = client.chat_postMessage(
    channel="#chatdsj_testing_ground",
    text="<@U08N3EFH6SE> What was a positive news story from today?"
)

print(f"Message sent: {response['ts']}")
