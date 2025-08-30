"""
llm_providers.py - LLM provider implementations with common interface
"""

import asyncio
import logging
import json
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from datetime import datetime
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold, GenerateContentConfig

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


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def analyze(self, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        """Analyze using OpenAI GPT"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert PostgreSQL performance engineer specializing in query optimization."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2500,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }

            timeout_config = aiohttp.ClientTimeout(total=timeout)

            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.post(self.api_url, headers=headers, json=data) as response:
                    result = await response.json()

                    if response.status == 200:
                        analysis = result["choices"][0]["message"]["content"]
                        metadata = {
                            "usage": result.get("usage", {}),
                            "finish_reason": result["choices"][0].get("finish_reason")
                        }
                        return self._create_result(True, analysis=analysis, metadata=metadata)
                    else:
                        error_msg = result.get("error", {}).get("message", f"API error: {response.status}")
                        logger.error(f"OpenAI API error: {error_msg}")
                        return self._create_result(False, error=error_msg)

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {timeout} seconds"
            logger.error(f"OpenAI timeout: {error_msg}")
            return self._create_result(False, error=error_msg)
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._create_result(False, error=str(e))

    def get_name(self) -> str:
        return "OpenAI"

    def get_model(self) -> str:
        return self.model


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation using the google.genai SDK"""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self.api_key = api_key
        self.model_name = model

        # Configure the client
        self.client = genai.Client(api_key=api_key)

    async def analyze(self, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        """Analyze using Google Gemini with the google.genai SDK"""

        try:
            # Prepare the generation configuration
            config = GenerateContentConfig(
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                # max_output_tokens=2500,
                stop_sequences=[],
            )

            # Run the generation in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            async def generate_with_timeout():
                return await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=[{"role": "user", "parts": [{"text": prompt}]}],
                        config=config
                    )
                )

            # Execute with timeout
            response = await asyncio.wait_for(
                generate_with_timeout(),
                timeout=timeout
            )

            # Process the response
            if response.candidates and response.candidates[0].content:
                content = response.candidates[0].content

                # Extract the text from parts
                text_parts = []
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                analysis = "".join(text_parts)

                if analysis:
                    # Extract metadata
                    candidate = response.candidates[0]
                    metadata = {
                        "finish_reason": candidate.finish_reason.name if hasattr(candidate,
                                                                                 'finish_reason') and candidate.finish_reason else None,
                        "safety_ratings": [
                            {
                                "category": rating.category.name,
                                "probability": rating.probability.name,
                                "blocked": getattr(rating, 'blocked', False)
                            }
                            for rating in (
                                candidate.safety_ratings
                                if hasattr(candidate,
                                           'safety_ratings') and
                                   candidate.safety_ratings is not None else [])
                        ],
                        "token_count": getattr(response, 'usage_metadata', {})
                    }

                    return self._create_result(True, analysis=analysis, metadata=metadata)
                else:
                    error_msg = "Generated response was empty"
                    return self._create_result(False, error=error_msg)
            else:
                # Handle blocked or failed generation
                if response.candidates and response.candidates[0]:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        error_msg = f"Response blocked: {candidate.finish_reason.name}"
                    else:
                        error_msg = "No content generated"
                else:
                    error_msg = "No response candidates received"

                logger.warning(f"Gemini response issue: {error_msg}")
                return self._create_result(False, error=error_msg)

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {timeout} seconds"
            logger.error(f"Gemini timeout: {error_msg}")
            return self._create_result(False, error=error_msg)

        except Exception as e:
            # Handle various SDK exceptions
            error_msg = str(e)

            # Categorize common error types
            if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                error_msg = "Invalid API key or authentication failed"
            elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                error_msg = "API quota exceeded or rate limited"
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                error_msg = f"Model '{self.model_name}' not found or not accessible"
            elif "permission" in error_msg.lower():
                error_msg = "Insufficient permissions to access the model"

            logger.error(f"Gemini analysis failed: {error_msg}")
            return self._create_result(False, error=error_msg)

    def get_name(self) -> str:
        return "Gemini"

    def get_model(self) -> str:
        return self.model_name

    async def list_available_models(self) -> list:
        """List available Gemini models"""
        try:
            loop = asyncio.get_event_loop()

            def get_models():
                return self.client.models.list()

            models_response = await loop.run_in_executor(None, get_models)

            # Filter for text generation models
            available_models = []
            for model in models_response.models:
                if hasattr(model, 'supported_generation_methods'):
                    if 'generateContent' in model.supported_generation_methods:
                        available_models.append(model.name)
                else:
                    # Fallback: include all models if we can't check methods
                    available_models.append(model.name)

            return available_models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def update_model(self, model: str):
        """Update the model being used"""
        self.model_name = model

    async def generate_stream(self, prompt: str, timeout: int = 60):
        """Generate streaming response (if supported by the SDK)"""
        try:
            config = GenerateContentConfig(
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                max_output_tokens=2500,
                stop_sequences=[],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            loop = asyncio.get_event_loop()

            def generate_stream_sync():
                return self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config=config
                )

            stream = await loop.run_in_executor(None, generate_stream_sync)
            return stream

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""

    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

    async def analyze(self, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        """Analyze using Anthropic Claude"""

        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2500,
                "temperature": 0.3,
                "top_p": 0.9,
                "system": "You are an expert PostgreSQL performance engineer specializing in query optimization, indexing strategies, and database tuning."
            }

            timeout_config = aiohttp.ClientTimeout(total=timeout)

            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.post(self.api_url, headers=headers, json=data) as response:
                    result = await response.json()

                    if response.status == 200:
                        analysis = result["content"][0]["text"]
                        metadata = {
                            "usage": result.get("usage", {}),
                            "stop_reason": result.get("stop_reason"),
                            "model": result.get("model")
                        }
                        return self._create_result(True, analysis=analysis, metadata=metadata)
                    else:
                        error_msg = result.get("error", {}).get("message", f"API error: {response.status}")
                        logger.error(f"Anthropic API error: {error_msg}")
                        return self._create_result(False, error=error_msg)

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {timeout} seconds"
            logger.error(f"Anthropic timeout: {error_msg}")
            return self._create_result(False, error=error_msg)
        except Exception as e:
            logger.error(f"Anthropic analysis failed: {e}")
            return self._create_result(False, error=str(e))

    def get_name(self) -> str:
        return "Anthropic"

    def get_model(self) -> str:
        return self.model


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

        if provider_name == "openai" and config.OPENAI_API_KEY:
            return OpenAIProvider(config.OPENAI_API_KEY, config.OPENAI_MODEL)

        elif provider_name == "gemini" and config.GEMINI_API_KEY:
            return GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL)

        elif provider_name == "anthropic" and config.ANTHROPIC_API_KEY:
            return AnthropicProvider(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        logger.warning(f"Provider {provider_name} not configured or not available")
        return None

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

        if config.OPENAI_API_KEY:
            providers["openai"] = OpenAIProvider(config.OPENAI_API_KEY, config.OPENAI_MODEL)

        if config.GEMINI_API_KEY:
            providers["gemini"] = GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL)

        if config.ANTHROPIC_API_KEY:
            providers["anthropic"] = AnthropicProvider(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        return providers
