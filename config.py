"""
config.py - Configuration management for the PostgreSQL Query Analyzer Bot
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables"""

    # Slack Configuration
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
    SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")

    # LLM Provider API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # OpenAI Model Configuration
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

    # Gemini Model Configuration
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

    # Anthropic Model Configuration
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")

    # Analysis Configuration
    TRIGGER_REACTIONS: List[str] = os.getenv(
        "TRIGGER_REACTIONS",
        "eyes,mag,rocket"
    ).split(",")

    DEFAULT_LLM_PROVIDERS: List[str] = os.getenv(
        "DEFAULT_LLM_PROVIDERS",
        "openai,gemini"
    ).split(",")

    ANALYSIS_TIMEOUT: int = int(os.getenv("ANALYSIS_TIMEOUT", "60"))

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Response Configuration
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "3000"))

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        errors = []

        if not cls.SLACK_BOT_TOKEN:
            errors.append("SLACK_BOT_TOKEN is required")

        if not cls.SLACK_APP_TOKEN:
            errors.append("SLACK_APP_TOKEN is required")

        # At least one LLM provider should be configured
        if not any([cls.OPENAI_API_KEY, cls.GEMINI_API_KEY, cls.ANTHROPIC_API_KEY]):
            errors.append("At least one LLM provider API key must be configured")

        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False

        return True


# Create global config instance
config = Config()
