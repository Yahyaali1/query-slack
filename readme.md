# PostgreSQL Query Analyzer Slack Bot

A Slack bot that automatically analyzes PostgreSQL queries and EXPLAIN outputs using multiple LLM providers (OpenAI, Google Gemini, Anthropic Claude) to provide optimization recommendations.

## Features

- 🔍 **Multi-LLM Analysis**: Compare insights from multiple AI providers
- ⚡ **Reaction-Based Triggers**: Simply react to a message to trigger analysis
- 📊 **Comprehensive Analysis**: Root cause analysis, index recommendations, query rewrites
- 🔄 **Concurrent Processing**: Analyze with multiple providers simultaneously
- 📝 **Actionable Output**: Copy-paste ready SQL statements for optimization
- 🛡️ **Error Handling**: Robust error handling and timeout management

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Slack     │────▶│  Slack Bot   │────▶│ Query Analyzer  │
│  Reaction   │     │   Handler    │     │   Orchestrator  │
└─────────────┘     └──────────────┘     └─────────────────┘
                            │                      │
                            ▼                      ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │   Response   │     │  LLM Providers  │
                    │  Formatter   │     │ • OpenAI        │
                    └──────────────┘     │ • Gemini        │
                                         │ • Anthropic     │
                                         └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Slack workspace with admin access
- At least one LLM provider API key (OpenAI, Gemini, or Anthropic)

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd pg-query-analyzer-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. Choose "From scratch" and give it a name (e.g., "PG Query Analyzer")
3. Select your workspace

#### Enable Socket Mode:
- Navigate to **Socket Mode** in the left sidebar
- Enable Socket Mode
- Generate an app-level token with `connections:write` scope
- Save the token (starts with `xapp-`)

#### Configure Bot Token Scopes:
- Go to **OAuth & Permissions**
- Add the following Bot Token Scopes:
  - `channels:history` - Read message history
  - `chat:write` - Send messages
  - `reactions:read` - Read reactions
  - `channels:read` - View basic channel info
  - `groups:history` - Read private channel history
  - `groups:read` - View private channels
  - `im:history` - Read direct message history
  - `mpim:history` - Read group DM history

#### Subscribe to Events:
- Go to **Event Subscriptions**
- Enable Events
- Subscribe to bot events:
  - `reaction_added`
  - `app_mention`

#### Install App:
- Go to **Install App**
- Click "Install to Workspace"
- Authorize the app
- Copy the Bot User OAuth Token (starts with `xoxb-`)

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required environment variables:
- `SLACK_BOT_TOKEN`: Your bot token (xoxb-...)
- `SLACK_APP_TOKEN`: Your app token (xapp-...)
- At least one LLM API key:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`
  - `ANTHROPIC_API_KEY`

### Step 4: Run the Bot

```bash
python main.py
```

You should see:
```
==================================================
PostgreSQL Query Analyzer Slack Bot
==================================================
Configuration Status:
  Slack Bot Token: ✓
  Slack App Token: ✓
  OpenAI API Key: ✓
  ...
Starting Slack bot...
```

## Usage

### Method 1: React to a Message

1. Post a message in Slack containing your query and EXPLAIN output:
```json
{
  "query": "SELECT * FROM orders WHERE status = 'pending' AND created_at > '2024-01-01'",
  "explain_output": "Seq Scan on orders (cost=0.00..50123.45 rows=10000 width=128)...",
  "execution_time": 5000,
  "database": "production"
}
```

2. React to the message with one of the trigger emojis:
   - 👀 (`:eyes:`)
   - 🔍 (`:mag:`)
   - 🚀 (`:rocket:`)

3. The bot will analyze the query and post results in a thread

### Method 2: Slash Command (if configured)

```
/analyze-query SELECT * FROM large_table WHERE ...
```

### Message Format Examples

#### JSON Format (Recommended):
```json
{
  "query": "SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id WHERE u.status = 'active'",
  "explain_output": "Nested Loop (cost=0.29..8.32 rows=1 width=8)\n  -> Seq Scan on users u (cost=0.00..4.00 rows=100 width=4)\n  -> Index Scan using orders_user_id_idx on orders o",
  "execution_time": 3500,
  "rows": 1000,
  "database": "analytics"
}
```

#### Plain Text Format:
```
Query: SELECT * FROM products WHERE category_id = 5
Execution Time: 2000ms
Database: shop_db

EXPLAIN ANALYZE:
Seq Scan on products (cost=0.00..1234.00 rows=50000 width=100)
  Filter: (category_id = 5)
  Rows Removed by Filter: 45000
```

## File Structure

```
pg-query-analyzer-bot/
├── config.py              # Configuration management
├── prompts.py             # LLM prompt templates
├── llm_providers.py       # LLM provider implementations
├── query_analyzer.py      # Analysis orchestrator
├── response_formatter.py  # Slack message formatting
├── slack_bot.py          # Slack event handlers
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── README.md            # Documentation
```

## Configuration Options

### Trigger Reactions
Customize which emoji reactions trigger analysis:
```bash
TRIGGER_REACTIONS=eyes,mag,rocket,zap,database
```

### LLM Providers
Choose which providers to use by default:
```bash
DEFAULT_LLM_PROVIDERS=openai,gemini,anthropic
```

### Timeout Settings
Adjust analysis timeout (seconds):
```bash
ANALYSIS_TIMEOUT=90
```

### Response Length
Maximum Slack message length:
```bash
MAX_RESPONSE_LENGTH=3000
```

## Example Output

```
📊 PostgreSQL Query Analysis Results
Generated at 2024-01-15 10:30:45 UTC

✅ 2 successful analyses

✅ OpenAI (gpt-4-turbo-preview)

Root Cause Analysis
The primary bottleneck is a sequential scan on the orders table with 1M+ rows...

Optimization Recommendations
Priority 1 - IMMEDIATE ACTIONS:
• CREATE INDEX idx_orders_status_date ON orders(status, created_at);
• ANALYZE orders;

Priority 2 - QUERY REFACTORING:
• Consider partitioning the orders table by created_at...

────────────────────────────────

✅ Gemini (gemini-pro)

Performance Issues Detected
1. Sequential scan on large table (line 2 of EXPLAIN)
2. Missing index on frequently filtered columns...

────────────────────────────────

📝 Extracted SQL Statements:
```
CREATE INDEX idx_orders_status_date ON orders(status, created_at);
ANALYZE orders;
```
```

## Troubleshooting

### Bot Not Responding to Reactions

1. Check Socket Mode is enabled in Slack app settings
2. Verify the bot is invited to the channel
3. Check logs for connection errors
4. Ensure reaction is in `TRIGGER_REACTIONS` list

### LLM Provider Errors

1. Verify API keys are correct
2. Check API quotas and rate limits
3. Ensure network connectivity
4. Review provider-specific error messages in logs

### Configuration Issues

Run configuration validation:
```python
python -c "from config import config; print(config.validate())"
```

## Security Considerations

- Never commit `.env` files to version control
- Use environment variables for all sensitive data
- Rotate API keys regularly
- Consider using secrets management services in production
- Implement rate limiting for Slack events

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add/update tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs for detailed error messages
3. Open an issue on GitHub
4. Contact your database team

## Roadmap

- [ ] Add support for MySQL and other databases
- [ ] Implement query performance tracking over time
- [ ] Add automated index application (with approval workflow)
- [ ] Create web dashboard for analysis history
- [ ] Add support for more LLM providers
- [ ] Implement cost tracking for LLM usage
- [ ] Add query pattern detection and learning