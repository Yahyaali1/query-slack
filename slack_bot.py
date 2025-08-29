"""
slack_bot.py - Slack bot implementation for query analysis
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from config import config
from query_analyzer import QueryAnalyzer
from response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class SlackBot:
    """Slack bot for PostgreSQL query analysis"""
    
    def __init__(self):
        """Initialize the Slack bot with configured handlers"""
        
        # Initialize Slack app
        self.app = AsyncApp(token=config.SLACK_BOT_TOKEN)
        
        # Initialize components
        self.analyzer = QueryAnalyzer()
        self.formatter = ResponseFormatter()
        
        # Track ongoing analyses to prevent duplicates
        self.ongoing_analyses = set()
        
        # Setup event handlers
        self._setup_handlers()
        
        logger.info("SlackBot initialized successfully")
    
    def _setup_handlers(self):
        """Setup Slack event handlers"""
        
        @self.app.event("reaction_added")
        async def handle_reaction_added(event, say, client, ack):
            """Handle reaction_added events"""
            
            # Acknowledge the event immediately
            await ack()
            
            # Check if reaction is one we care about
            reaction = event.get("reaction", "")
            if reaction not in config.TRIGGER_REACTIONS:
                logger.debug(f"Ignoring reaction: {reaction}")
                return
            
            # Extract event details
            user = event.get("user", "")
            channel = event.get("item", {}).get("channel", "")
            timestamp = event.get("item", {}).get("ts", "")
            
            # Create unique key for this analysis
            analysis_key = f"{channel}:{timestamp}:{reaction}"
            
            # Check if analysis is already in progress
            if analysis_key in self.ongoing_analyses:
                logger.info(f"Analysis already in progress for {analysis_key}")
                return
            
            # Mark analysis as ongoing
            self.ongoing_analyses.add(analysis_key)
            
            try:
                await self._process_reaction(
                    client=client,
                    channel=channel,
                    timestamp=timestamp,
                    user=user,
                    reaction=reaction
                )
            except Exception as e:
                logger.error(f"Error processing reaction: {e}", exc_info=True)
                await self._send_error_message(client, channel, timestamp, str(e))
            finally:
                # Remove from ongoing analyses
                self.ongoing_analyses.discard(analysis_key)
        
        @self.app.event("app_mention")
        async def handle_app_mention(event, say, ack):
            """Handle app mentions"""
            
            await ack()
            
            text = event.get("text", "")
            channel = event.get("channel", "")
            
            # Simple help response
            if "help" in text.lower():
                await say(
                    channel=channel,
                    text=self._get_help_message()
                )
            else:
                await say(
                    channel=channel,
                    text=(
                        "👋 Hi! React to a query message with one of these emojis to analyze it:\n"
                        f"• {', '.join([f':{r}:' for r in config.TRIGGER_REACTIONS])}"
                    )
                )
        
        @self.app.command("/analyze-query")
        async def handle_analyze_command(ack, command, client):
            """Handle /analyze-query slash command"""
            
            await ack()
            
            # Parse command text as query data
            query_data = self.analyzer.parse_slack_message(command["text"])
            
            # Send initial message
            response = await client.chat_postMessage(
                channel=command["channel_id"],
                text=f"📊 Analyzing query from {command['user_name']}..."
            )
            
            # Perform analysis
            results = await self.analyzer.analyze_with_multiple_providers(query_data)
            
            # Update with results
            formatted_response = self.formatter.format_analysis_results(results)
            
            await client.chat_update(
                channel=command["channel_id"],
                ts=response["ts"],
                text=formatted_response
            )
    
    async def _process_reaction(
        self,
        client,
        channel: str,
        timestamp: str,
        user: str,
        reaction: str
    ):
        """
        Process a reaction event and trigger analysis
        
        Args:
            client: Slack client
            channel: Channel ID
            timestamp: Message timestamp
            user: User who added reaction
            reaction: Reaction emoji
        """
        
        logger.info(f"Processing reaction {reaction} from {user} on {channel}:{timestamp}")
        
        # Fetch the original message
        try:
            result = await client.conversations_history(
                channel=channel,
                latest=timestamp,
                limit=1,
                inclusive=True
            )
            
            if not result.get("messages"):
                logger.error("Could not find original message")
                await self._send_error_message(
                    client, channel, timestamp, 
                    "Could not find the original message"
                )
                return
            
            original_message = result["messages"][0]
            message_text = original_message.get("text", "")
            
        except Exception as e:
            logger.error(f"Failed to fetch message: {e}")
            await self._send_error_message(
                client, channel, timestamp,
                f"Failed to fetch message: {str(e)}"
            )
            return
        
        # Parse query data from message
        query_data = self.analyzer.parse_slack_message(message_text)
        
        # Check if we have minimum required data
        if not query_data.get("query"):
            await client.chat_postMessage(
                channel=channel,
                thread_ts=timestamp,
                text=(
                    "⚠️ No SQL query found in the message. "
                    "Please ensure the message contains a query to analyze."
                )
            )
            return
        
        # Get available providers
        provider_status = self.analyzer.get_provider_status()
        available_count = provider_status["total_configured"]
        
        if available_count == 0:
            await client.chat_postMessage(
                channel=channel,
                thread_ts=timestamp,
                text="❌ No LLM providers are configured. Please check your environment variables."
            )
            return
        
        # Send initial acknowledgment
        ack_message = self.formatter.format_acknowledgment(available_count)
        
        initial_response = await client.chat_postMessage(
            channel=channel,
            thread_ts=timestamp,
            text=ack_message
        )
        
        # Perform analysis with multiple providers
        try:
            results = await self.analyzer.analyze_with_multiple_providers(query_data)
            
            # Format results
            formatted_response = self.formatter.format_analysis_results(results)
            
            # Update the initial message with results
            await client.chat_update(
                channel=channel,
                ts=initial_response["ts"],
                text=formatted_response
            )
            
            # Extract and post SQL statements separately if found
            sql_statements = []
            for result in results:
                if result.get("success") and result.get("analysis"):
                    statements = self.analyzer.extract_sql_statements(result["analysis"])
                    sql_statements.extend(statements)
            
            if sql_statements:
                # Remove duplicates while preserving order
                unique_statements = []
                seen = set()
                for stmt in sql_statements:
                    if stmt not in seen:
                        seen.add(stmt)
                        unique_statements.append(stmt)
                
                # Post SQL statements in thread
                sql_formatted = self.formatter.format_sql_statements(unique_statements)
                await client.chat_postMessage(
                    channel=channel,
                    thread_ts=timestamp,
                    text=sql_formatted
                )
            
            logger.info(f"Successfully completed analysis for {channel}:{timestamp}")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            
            # Update message with error
            await client.chat_update(
                channel=channel,
                ts=initial_response["ts"],
                text=self.formatter.format_error_message(str(e))
            )
    
    async def _send_error_message(
        self,
        client,
        channel: str,
        thread_ts: str,
        error: str
    ):
        """Send error message to Slack thread"""
        
        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=self.formatter.format_error_message(error)
        )
    
    def _get_help_message(self) -> str:
        """Get help message for users"""
        
        return f"""
*PostgreSQL Query Analyzer Bot - Help*

*How to use:*
1. Post a message containing a slow query and its EXPLAIN output
2. React to the message with one of these emojis: {', '.join([f':{r}:' for r in config.TRIGGER_REACTIONS])}
3. The bot will analyze the query using multiple LLM providers
4. Results will appear in a thread with optimization recommendations

*Message Format:*
Your message should contain:
• The SQL query (in a code block or plain text)
• EXPLAIN ANALYZE output
• Execution time (optional)
• Database name (optional)

*Example:*
```
Query: SELECT * FROM users WHERE status = 'active'
Execution Time: 5000ms
EXPLAIN: Seq Scan on users (cost=0.00..1234.00 rows=50000)
```

*Available Providers:*
{self._get_provider_status()}

*Support:*
For issues or questions, contact your database team.
        """
    
    def _get_provider_status(self) -> str:
        """Get formatted provider status"""
        
        status = self.analyzer.get_provider_status()
        
        providers = []
        if status.get("openai"):
            providers.append("• OpenAI GPT ✅")
        if status.get("gemini"):
            providers.append("• Google Gemini ✅")
        if status.get("anthropic"):
            providers.append("• Anthropic Claude ✅")
        
        if not providers:
            return "❌ No providers configured"
        
        return "\n".join(providers)
    
    async def start(self):
        """Start the Slack bot"""
        
        try:
            # Validate configuration
            if not config.validate():
                logger.error("Configuration validation failed")
                return
            
            logger.info("Starting Slack bot...")
            
            # Create and start socket mode handler
            handler = AsyncSocketModeHandler(self.app, config.SLACK_APP_TOKEN)
            await handler.start_async()
            
        except Exception as e:
            logger.error(f"Failed to start Slack bot: {e}", exc_info=True)
            raise