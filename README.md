# ChatDSJ Slackbot

A modern Slackbot with natural language processing capabilities, built with a modular architecture for maintainability and scalability.

## Features

*   Responds intelligently when mentioned in Slack channels
*   Contextual awareness of conversation history
*   Personalized responses based on user preferences stored in Notion
*   Web content and YouTube video summarization
*   Task management with TODO tracking
*   Memory capabilities for user facts and preferences

## Architecture

The application is built with a modular, service-oriented architecture:

*   **Services:** Encapsulate external API interactions (Slack, Notion, OpenAI)
*   **Handlers:** Manage business logic for specific domains
*   **Actions:** Implement specific user-triggered behaviors
*   **Utils:** Provide shared functionality across the application

## Setup

1.  **Install dependencies:**

    ```bash
    poetry install
    ```

2.  **Create a Slack app at [https://api.slack.com/apps](https://api.slack.com/apps)**
    *   Choose the "Create from App Manifest" option.
    *   Paste the following manifest JSON:

        ```json
        {
            "display_information": {
                "name": "ChatDSJ Bot",
                "description": "A Slackbot with enhanced capabilities",
                "background_color": "#2c2d30"
            },
            "features": {
                "bot_user": {
                    "display_name": "ChatDSJ Bot",
                    "always_online": true
                },
                "app_home": {
                    "home_tab_enabled": false,
                    "messages_tab_enabled": true,
                    "messages_tab_read_only_enabled": false
                }
            },
            "oauth_config": {
                "scopes": {
                    "bot": [
                        "app_mentions:read",
                        "channels:history",
                        "chat:write",
                        "users:read",
                        "groups:history",
                        "reactions:read",
                        "reactions:write"
                    ]
                }
            },
            "settings": {
                "event_subscriptions": {
                    "bot_events": [
                        "app_mention"
                    ]
                },
                "interactivity": {
                    "is_enabled": false
                },
                "org_deploy_enabled": false,
                "socket_mode_enabled": true,
                "token_rotation_enabled": false
            }
        }
        ```

    *   After creating the app:
        *   Go to "Basic Information" to get your **Signing Secret**.
        *   Go to "OAuth & Permissions" to get your **Bot Token** (starts with `xoxb-`).
        *   Go to "Socket Mode" and enable it, then generate an **App-Level Token** (starts with `xapp-`).
    *   Install the app to your workspace.

3.  **Set up a Notion integration (optional but recommended):**
    *   Create a new integration at [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations).
    *   Create a database for user profiles.
    *   Share the database with your integration.
    *   Note the database ID from the URL.

4.  **Create a `.env` file with your credentials:**

    ```
    # Slack Configuration
    SLACK_BOT_TOKEN=xoxb-your-token
    SLACK_SIGNING_SECRET=your-signing-secret
    SLACK_APP_TOKEN=xapp-your-app-token

    # OpenAI Configuration
    OPENAI_API_KEY=your-openai-api-key
    OPENAI_MODEL=gpt-4o

    # Notion Configuration (optional)
    NOTION_API_TOKEN=secret_your-notion-api-token
    NOTION_USER_DB_ID=your-notion-database-id

    # Application Configuration (Optional)
    LOG_LEVEL=INFO
    ENVIRONMENT=development
    MAX_TOKENS_RESPONSE=1500
    MAX_MESSAGE_HISTORY=1000

    # Caching Configuration (Optional)
    CACHE_TTL=300
    CACHE_MAX_SIZE=1000
    ```

5.  **Run the server:**

    ```bash
    poetry run uvicorn main:app --reload
    ```

6.  **Test the bot:**
    *   Invite the bot to a channel in your Slack workspace.
    *   Mention the bot with `@YourBotName`.
    *   The bot will respond based on the conversation context.

## Deployment

The application is configured for deployment to Fly.io:

1.  **Install the Fly CLI:** [https://fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)
2.  **Authenticate and create an app:**

    ```bash
    flyctl auth login
    flyctl launch
    ```

3.  **Deploy manually:**

    ```bash
    flyctl deploy
    ```

4.  Or use the GitHub Action workflow for automatic deployment on push to main branch.

## Development

### Project Structure

    root/
    ├── actions/          # Action handlers for different user interactions
    ├── config/           # Configuration management
    ├── handler/          # Business logic handlers
    ├── services/         # External service integrations
    ├── tests/            # Unit and integration tests
    ├── utils/            # Shared utilities
    ├── main.py           # Application entry point
    ├── Dockerfile        # Container definition
    └── pyproject.toml    # Dependencies and project metadata


### Running Tests

    poetry run python -m unittest discover -s tests


### Adding New Capabilities

*   New Service: Add to the `services/` directory and register in `ServiceContainer`
*   New Action: Extend the `Action` class in `actions/action_framework.py`
*   New Utility: Add to the `utils/` directory for shared functionality

## License

[Your License Here]