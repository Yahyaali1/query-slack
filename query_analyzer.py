"""
query_analyzer.py - Main query analysis orchestrator
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import config
from prompts import PromptTemplates
from llm_providers import LLMProviderFactory

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Orchestrates query analysis across multiple LLM providers"""
    
    def __init__(self):
        self.providers = LLMProviderFactory.get_available_providers(config)
        self.prompt_templates = PromptTemplates()
        logger.info(f"Initialized QueryAnalyzer with providers: {list(self.providers.keys())}")
    
    def parse_slack_message(self, message: str) -> Dict[str, Any]:
        """
        Parse query data from Slack message
        
        Args:
            message: Raw Slack message text
            
        Returns:
            Dictionary containing parsed query data
        """
        
        try:
            # Try to find JSON block in message
            if "```json" in message:
                json_start = message.index("```json") + 7
                json_end = message.index("```", json_start)
                json_str = message[json_start:json_end].strip()
                return json.loads(json_str)
            
            # Try to find plain JSON (look for { and })
            elif "{" in message and "}" in message:
                json_start = message.index("{")
                json_end = message.rindex("}") + 1
                json_str = message[json_start:json_end]
                return json.loads(json_str)
            
            # Try to extract SQL and EXPLAIN separately
            else:
                return self._parse_structured_message(message)
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse message as JSON: {e}")
            return self._parse_fallback(message)
    
    def _parse_structured_message(self, message: str) -> Dict[str, Any]:
        """
        Parse structured message with SQL and EXPLAIN sections
        
        Args:
            message: Message text
            
        Returns:
            Parsed data dictionary
        """
        
        data = {
            "query": "",
            "explain_output": "",
            "execution_time": "Unknown",
            "database": "Unknown"
        }
        
        # Extract SQL query
        sql_patterns = [
            r"```sql\n(.*?)```",
            r"Query:\s*\n(.*?)(?:\n\n|EXPLAIN|$)",
            r"SQL:\s*\n(.*?)(?:\n\n|EXPLAIN|$)"
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
            if match:
                data["query"] = match.group(1).strip()
                break
        
        # Extract EXPLAIN output
        explain_patterns = [
            r"EXPLAIN.*?:\s*\n(.*?)(?:```|$)",
            r"```\n(.*?)```",
            r"Plan:\s*\n(.*?)(?:\n\n|$)"
        ]
        
        for pattern in explain_patterns:
            match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
            if match:
                data["explain_output"] = match.group(1).strip()
                break
        
        # Extract execution time
        time_match = re.search(r"(?:execution time|runtime|time):\s*(\d+(?:\.\d+)?)\s*ms", 
                              message, re.IGNORECASE)
        if time_match:
            data["execution_time"] = time_match.group(1)
        
        # Extract database name
        db_match = re.search(r"(?:database|db):\s*(\w+)", message, re.IGNORECASE)
        if db_match:
            data["database"] = db_match.group(1)
        
        return data
    
    def _parse_fallback(self, message: str) -> Dict[str, Any]:
        """
        Fallback parser - treat entire message as query
        
        Args:
            message: Message text
            
        Returns:
            Basic data dictionary
        """
        
        logger.info("Using fallback parser for message")
        
        return {
            "query": message,  # Limit length
            "explain_output": "Not provided",
            "execution_time": "Unknown",
            "database": "Unknown",
            "parse_method": "fallback"
        }
    
    async def analyze_with_provider(
        self, 
        provider_name: str, 
        query_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze query with a specific provider
        
        Args:
            provider_name: Name of the provider to use
            query_data: Parsed query data
            
        Returns:
            Analysis result dictionary
        """
        
        if provider_name not in self.providers:
            logger.warning(f"Provider {provider_name} not available")
            return {
                "success": False,
                "provider": provider_name,
                "error": f"Provider {provider_name} not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        provider = self.providers[provider_name]
        prompt = self.prompt_templates.get_analysis_prompt(query_data)
        
        logger.info(f"Starting analysis with {provider_name}")
        
        try:
            result = await provider.analyze(prompt, timeout=config.ANALYSIS_TIMEOUT)
            logger.info(f"Completed analysis with {provider_name}: success={result.get('success')}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {provider_name}: {e}")
            return {
                "success": False,
                "provider": provider_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def analyze_with_multiple_providers(
        self, 
        query_data: Dict[str, Any],
        provider_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze query with multiple providers concurrently
        
        Args:
            query_data: Parsed query data
            provider_names: List of provider names to use (None = use defaults)
            
        Returns:
            List of analysis results from all providers
        """
        
        if provider_names is None:
            provider_names = config.DEFAULT_LLM_PROVIDERS
        
        # Filter to only available providers
        available_providers = [p for p in provider_names if p in self.providers]
        
        if not available_providers:
            logger.error("No LLM providers available")
            return [{
                "success": False,
                "error": "No LLM providers configured",
                "timestamp": datetime.utcnow().isoformat()
            }]
        
        logger.info(f"Starting concurrent analysis with providers: {available_providers}")
        
        # Create analysis tasks
        tasks = [
            self.analyze_with_provider(provider, query_data)
            for provider in available_providers
        ]
        
        # Run analyses concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Provider {available_providers[i]} raised exception: {result}")
                processed_results.append({
                    "success": False,
                    "provider": available_providers[i],
                    "error": str(result),
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                processed_results.append(result)
        
        logger.info(f"Completed analysis with {len(processed_results)} providers")
        return processed_results
    
    async def compare_analyses(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare and synthesize multiple analysis results
        
        Args:
            results: List of analysis results from different providers
            
        Returns:
            Synthesized comparison result
        """
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {
                "success": False,
                "error": "No successful analyses to compare",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Use the comparison prompt if we have multiple successful results
        if len(successful_results) > 1:
            comparison_prompt = self.prompt_templates.get_comparison_prompt(successful_results)
            
            # Use the first available provider for comparison
            provider_name = list(self.providers.keys())[0]
            provider = self.providers[provider_name]
            
            try:
                comparison_result = await provider.analyze(comparison_prompt, timeout=30)
                return {
                    "success": True,
                    "comparison": comparison_result.get("analysis", ""),
                    "provider_count": len(successful_results),
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Comparison analysis failed: {e}")
                return {
                    "success": False,
                    "error": f"Comparison failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
        else:
            # Only one successful result, no comparison needed
            return {
                "success": True,
                "comparison": "Only one provider succeeded, no comparison available",
                "provider_count": 1,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def extract_sql_statements(self, analysis: str) -> List[str]:
        """
        Extract SQL statements from analysis text
        
        Args:
            analysis: Analysis text containing SQL statements
            
        Returns:
            List of SQL statements
        """
        
        sql_statements = []
        
        # Pattern to match SQL blocks
        sql_block_pattern = r"```sql\n(.*?)```"
        matches = re.findall(sql_block_pattern, analysis, re.DOTALL)
        
        for match in matches:
            # Clean and split multiple statements
            statements = match.strip().split(';')
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith('--'):
                    sql_statements.append(stmt + ';')
        
        # Also look for inline CREATE INDEX statements
        index_pattern = r"(CREATE\s+(?:UNIQUE\s+)?INDEX\s+.*?(?:;|\n|$))"
        inline_matches = re.findall(index_pattern, analysis, re.IGNORECASE | re.DOTALL)
        
        for match in inline_matches:
            if match not in sql_statements:
                sql_statements.append(match.strip())
        
        return sql_statements
    
    def get_provider_status(self) -> Dict[str, bool]:
        """
        Get the configuration status of all providers
        
        Returns:
            Dictionary with provider availability status
        """
        
        return {
            "openai": "openai" in self.providers,
            "gemini": "gemini" in self.providers,
            "anthropic": "anthropic" in self.providers,
            "total_configured": len(self.providers)
        }