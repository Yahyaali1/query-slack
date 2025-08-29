"""
main.py - Simplified main entry point for Gemini-only PostgreSQL Query Analyzer
"""

import asyncio
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from llm_providers import QueryAnalyzer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class SimpleQueryAnalyzerBot:
    """Simplified bot for PostgreSQL query analysis using Gemini"""

    def __init__(self):
        """Initialize the bot with Gemini configuration"""

        # Get configuration from environment
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY is required")

        # Initialize the analyzer
        self.analyzer = QueryAnalyzer(self.api_key, self.model)
        logger.info(f"Bot initialized with model: {self.model}")

    async def analyze_from_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a query from text input

        Args:
            text: Input text containing query and EXPLAIN data

        Returns:
            Analysis result
        """

        # Parse the input
        query_data = self.analyzer.parse_query_data(text)

        # Validate we have minimum required data
        if not query_data.get("query"):
            return {
                "success": False,
                "error": "No SQL query found in the input"
            }

        # Analyze the query
        logger.info("Starting query analysis...")
        result = await self.analyzer.analyze_query(query_data)

        return result

    async def interactive_mode(self):
        """Run in interactive mode for testing"""

        print("\n" + "="*60)
        print("PostgreSQL Query Analyzer - Interactive Mode")
        print("Using Gemini model:", self.model)
        print("="*60)
        print("\nPaste your query and EXPLAIN output, then type 'ANALYZE' on a new line:")
        print("(Type 'EXIT' to quit)\n")

        while True:
            try:
                # Collect multi-line input
                lines = []
                while True:
                    line = input()
                    if line.upper() == "ANALYZE":
                        break
                    if line.upper() == "EXIT":
                        print("Goodbye!")
                        return
                    lines.append(line)

                if not lines:
                    print("No input provided. Try again.\n")
                    continue

                # Join the input
                input_text = "\n".join(lines)

                # Analyze
                print("\n🔍 Analyzing query with Gemini...\n")
                result = await self.analyze_from_text(input_text)

                # Display result
                if result.get("success"):
                    print("✅ Analysis Complete:\n")
                    print(result.get("analysis", "No analysis provided"))

                    # Show thinking process if available
                    if result.get("thinking_process"):
                        print("\n💭 Thinking Process:")
                        print(result["thinking_process"])
                else:
                    print("❌ Analysis Failed:")
                    print(result.get("error", "Unknown error"))

                print("\n" + "-"*60)
                print("\nPaste another query or type 'EXIT' to quit:\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {e}")
                print("Try again or type 'EXIT' to quit.\n")


async def main():
    """Main entry point"""

    try:
        # Check if we have example input file
        if len(sys.argv) > 1:
            # File mode
            filename = sys.argv[1]

            if not os.path.exists(filename):
                print(f"Error: File '{filename}' not found")
                sys.exit(1)

            with open(filename, 'r') as f:
                input_text = f.read()

            bot = SimpleQueryAnalyzerBot()
            result = await bot.analyze_from_text(input_text)

            if result.get("success"):
                print("Analysis Result:")
                print("="*60)
                print(result.get("analysis", "No analysis provided"))
            else:
                print(f"Analysis failed: {result.get('error', 'Unknown error')}")

        else:
            # Interactive mode
            bot = SimpleQueryAnalyzerBot()
            await bot.interactive_mode()

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        logger.error(f"Failed to run application: {e}")
        sys.exit(1)