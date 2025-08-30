"""
main.py - Main entry point for the PostgreSQL Query Analyzer Slack Bot
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from config import config
from slack_bot import SlackBot

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Optionally add file handler
        # logging.FileHandler('query_analyzer.log')
    ]
)

logger = logging.getLogger(__name__)

class Application:
    """Main application class"""

    def __init__(self):
        self.bot: Optional[SlackBot] = None
        self.shutdown_event = asyncio.Event()

    async def start(self):
        """Start the application"""

        logger.info("=" * 50)
        logger.info("PostgreSQL Query Analyzer Slack Bot")
        logger.info("=" * 50)

        # Log configuration status
        self._log_configuration()

        # Validate configuration
        if not config.validate():
            logger.error("Configuration validation failed. Exiting.")
            sys.exit(1)

        # Initialize and start the bot
        try:
            logger.info("Initializing Slack bot...")
            self.bot = SlackBot()

            logger.info("Starting Slack bot...")

            # Start bot in background
            bot_task = asyncio.create_task(self.bot.start())

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            logger.info("Shutting down...")
            bot_task.cancel()

            try:
                await bot_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            sys.exit(1)

    def _log_configuration(self):
        """Log current configuration status"""

        logger.info("Configuration Status:")
        logger.info(f"  Slack Bot Token: {'✓' if config.SLACK_BOT_TOKEN else '✗'}")
        logger.info(f"  Slack App Token: {'✓' if config.SLACK_APP_TOKEN else '✗'}")
        logger.info(f"  OpenAI API Key: {'✓' if config.OPENAI_API_KEY else '✗'}")
        logger.info(f"  Gemini API Key: {'✓' if config.GEMINI_API_KEY else '✗'}")
        logger.info(f"  Anthropic API Key: {'✓' if config.ANTHROPIC_API_KEY else '✗'}")
        logger.info(f"  Trigger Reactions: {', '.join(config.TRIGGER_REACTIONS)}")
        logger.info(f"  Default Providers: {', '.join(config.DEFAULT_LLM_PROVIDERS)}")
        logger.info(f"  Analysis Timeout: {config.ANALYSIS_TIMEOUT}s")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""

        logger.info(f"Received signal {signum}. Initiating shutdown...")
        self.shutdown_event.set()

async def main():
    """Main function"""

    app = Application()

    # Setup signal handlers
    signal.signal(signal.SIGINT, app.handle_shutdown)
    signal.signal(signal.SIGTERM, app.handle_shutdown)

    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Failed to run application: {e}", exc_info=True)
        sys.exit(1)