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
    """Google Gemini provider implementation"""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def analyze(self, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        """Analyze using Google Gemini"""

        try:
            headers = {
                "Content-Type": "application/json"
            }

            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 40,
                    "topP": 0.9,
                    "maxOutputTokens": 2500,
                    "stopSequences": []
                },
                "safetySettings": [
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
            }

            url = f"{self.api_url}?key={self.api_key}"
            timeout_config = aiohttp.ClientTimeout(total=timeout)

            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    result = await response.json()

                    if response.status == 200:
                        if "candidates" in result and result["candidates"]:
                            analysis = result["candidates"][0]["content"]["parts"][0]["text"]
                            metadata = {
                                "finish_reason": result["candidates"][0].get("finishReason"),
                                "safety_ratings": result["candidates"][0].get("safetyRatings", [])
                            }
                            return self._create_result(True, analysis=analysis, metadata=metadata)
                        else:
                            error_msg = "No response generated"
                            return self._create_result(False, error=error_msg)
                    else:
                        error_msg = result.get("error", {}).get("message", f"API error: {response.status}")
                        logger.error(f"Gemini API error: {error_msg}")
                        return self._create_result(False, error=error_msg)

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {timeout} seconds"
            logger.error(f"Gemini timeout: {error_msg}")
            return self._create_result(False, error=error_msg)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self._create_result(False, error=str(e))

    def get_name(self) -> str:
        return "Gemini"

    def get_model(self) -> str:
        return self.model


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