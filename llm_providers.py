"""
llm_providers.py - LLM provider implementations with common interface
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def analyze(self, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Analyze query using the LLM

        Args:
            prompt: The analysis prompt
            timeout: Request timeout in seconds

        Returns:
            Dictionary with analysis results
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name"""
        pass

    @abstractmethod
    def get_model(self) -> str:
        """Get model name"""
        pass

    def _create_result(
        self,
        success: bool,
        analysis: str = None,
        error: str = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """Create standardized result dictionary"""

        result = {
            "success": success,
            "provider": self.get_name(),
            "model": self.get_model(),
            "timestamp": datetime.utcnow().isoformat()
        }

        if success and analysis:
            result["analysis"] = analysis
        elif error:
            result["error"] = error

        if metadata:
            result["metadata"] = metadata

        return result




class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation using google-generativeai library"""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai library is required for Gemini provider. "
                "Install it with: pip install google-generativeai"
            )

        # Configure the API key
        self.genai.configure(api_key=api_key)

        # Initialize the model with custom configuration
        self.model_name = model

        # Model configuration for PostgreSQL analysis
        generation_config = {
            "temperature": 0.3,  # Lower for more focused responses
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2500,
            "stop_sequences": [],
        }

        # Safety settings - set to most permissive for technical content
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        # Initialize the model
        self.model = self.genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction="You are an expert PostgreSQL database performance engineer with deep expertise in query optimization, indexing strategies, execution plan analysis, and database tuning. Provide detailed, actionable recommendations with specific SQL statements."
        )

        logger.info(f"Initialized Gemini provider with model: {model}")

    async def analyze(self, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        """Analyze using Google Gemini with async execution"""

        try:
            # Gemini SDK doesn't have native async, so we run it in executor
            loop = asyncio.get_event_loop()

            # Create a wrapper function for timeout handling
            async def generate_with_timeout():
                return await loop.run_in_executor(
                    None,
                    self._generate_content,
                    prompt
                )

            # Apply timeout
            response = await asyncio.wait_for(
                generate_with_timeout(),
                timeout=timeout
            )

            # Check if response is valid
            if response and response.text:
                metadata = self._extract_metadata(response)

                return self._create_result(
                    True,
                    analysis=response.text,
                    metadata=metadata
                )
            else:
                # Handle blocked or empty responses
                if response and hasattr(response, 'prompt_feedback'):
                    error_msg = f"Response blocked: {response.prompt_feedback}"
                else:
                    error_msg = "No response generated from Gemini"

                logger.warning(f"Gemini empty response: {error_msg}")
                return self._create_result(False, error=error_msg)

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {timeout} seconds"
            logger.error(f"Gemini timeout: {error_msg}")
            return self._create_result(False, error=error_msg)

        except Exception as e:
            error_msg = self._handle_gemini_error(e)
            logger.error(f"Gemini analysis failed: {error_msg}")
            return self._create_result(False, error=error_msg)

    def _generate_content(self, prompt: str):
        """Synchronous content generation wrapper"""
        try:
            # You can also use generate_content_async if available in newer versions
            response = self.model.generate_content(
                prompt,
                stream=False  # Don't stream for simpler handling
            )
            return response
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise

    def _extract_metadata(self, response) -> Dict[str, Any]:
        """Extract metadata from Gemini response"""
        metadata = {
            "model": self.model_name
        }

        # Extract safety ratings if available
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]

            if hasattr(candidate, 'safety_ratings'):
                metadata["safety_ratings"] = [
                    {
                        "category": str(rating.category),
                        "probability": str(rating.probability)
                    }
                    for rating in candidate.safety_ratings
                ]

            if hasattr(candidate, 'finish_reason'):
                metadata["finish_reason"] = str(candidate.finish_reason)

        # Extract token count if available
        if hasattr(response, 'usage_metadata'):
            metadata["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }

        return metadata

    def _handle_gemini_error(self, error: Exception) -> str:
        """Handle and format Gemini-specific errors"""
        error_str = str(error)

        # Common Gemini error patterns
        if "quota" in error_str.lower():
            return "API quota exceeded. Please check your Gemini API limits."
        elif "api key" in error_str.lower():
            return "Invalid API key. Please check your Gemini API key configuration."
        elif "safety" in error_str.lower():
            return "Response blocked by safety filters. This shouldn't happen for technical content."
        elif "resource exhausted" in error_str.lower():
            return "Rate limit exceeded. Please wait before retrying."
        elif "invalid argument" in error_str.lower():
            return f"Invalid request parameters: {error_str}"
        else:
            return f"Gemini API error: {error_str}"

    def get_name(self) -> str:
        return "Gemini"

    def get_model(self) -> str:
        return self.model_name

    def get_available_models(self) -> list:
        """Get list of available Gemini models"""
        try:
            models = self.genai.list_models()
            return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except Exception as e:
            logger.error(f"Failed to list Gemini models: {e}")
            return [self.model_name]

class LLMProviderFactory:
    """Factory class for creating LLM providers"""

    @staticmethod
    def create_provider(provider_name: str, config: Any) -> Optional[LLMProvider]:
        """
        Create an LLM provider instance

        Args:
            provider_name: Name of the provider (openai, gemini, anthropic)
            config: Configuration object with API keys

        Returns:
            LLMProvider instance or None if not configured
        """

        provider_name = provider_name.lower()


        return GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL)

    @staticmethod
    def get_available_providers(config: Any) -> Dict[str, LLMProvider]:
        """
        Get all available providers based on configuration

        Args:
            config: Configuration object with API keys

        Returns:
            Dictionary of available providers
        """

        providers = {}

        if config.GEMINI_API_KEY:
            providers["gemini"] = GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL)

        return providers