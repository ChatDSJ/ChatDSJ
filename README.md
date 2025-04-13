# ChatDSJ Slackbot

A Python backend server that integrates with Slack.

## Features

- Responds when mentioned in a Slack channel
- Uses randomly selected rude phrases for responses
- Tracks message counts and participants in channels

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Create a Slack app at https://api.slack.com/apps
   - Add the following bot token scopes:
     - `app_mentions:read`
     - `channels:history`
     - `chat:write`
     - `users:read`
   - Install the app to your workspace
   - Copy the Bot Token and Signing Secret

3. Create a `.env` file with your Slack credentials:
```
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_SIGNING_SECRET=your-signing-secret
```

4. Run the server:
```bash
poetry run uvicorn app.main:app --reload
```
