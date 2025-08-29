"""
response_formatter.py - Format LLM responses for Slack display
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import config

class ResponseFormatter:
    """Format analysis results for Slack display"""
    
    def __init__(self):
        self.max_length = config.MAX_RESPONSE_LENGTH
    
    def format_analysis_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format multiple provider results for Slack
        
        Args:
            results: List of analysis results from providers
            
        Returns:
            Formatted string for Slack message
        """
        
        if not results:
            return "❌ No analysis results available"
        
        # Separate successful and failed results
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        # Start building the response
        formatted = self._create_header(len(successful), len(failed))
        
        # Add successful analyses
        if successful:
            for result in successful:
                formatted += self._format_successful_analysis(result)
                formatted += "\n" + "─" * 40 + "\n\n"
        
        # Add failed analyses at the end
        if failed:
            formatted += "\n*Failed Analyses:*\n"
            for result in failed:
                formatted += self._format_failed_analysis(result)
        
        # Add footer with metadata
        formatted += self._create_footer(results)
        
        return formatted
    
    def format_single_result(self, result: Dict[str, Any]) -> str:
        """
        Format a single analysis result
        
        Args:
            result: Single analysis result
            
        Returns:
            Formatted string for Slack
        """
        
        if result.get("success", False):
            return self._format_successful_analysis(result)
        else:
            return self._format_failed_analysis(result)
    
    def format_comparison_result(
        self, 
        comparison: Dict[str, Any], 
        original_results: List[Dict[str, Any]]
    ) -> str:
        """
        Format comparison/synthesis of multiple analyses
        
        Args:
            comparison: Comparison analysis result
            original_results: Original analysis results
            
        Returns:
            Formatted comparison for Slack
        """
        
        formatted = "🔄 *Comparative Analysis*\n\n"
        
        if comparison.get("success", False):
            formatted += comparison.get("comparison", "No comparison available")
        else:
            formatted += f"❌ Comparison failed: {comparison.get('error', 'Unknown error')}"
        
        formatted += f"\n\n_Based on {comparison.get('provider_count', 0)} provider analyses_"
        
        return formatted
    
    def _create_header(self, success_count: int, failed_count: int) -> str:
        """Create header for the response"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        header = f"📊 *PostgreSQL Query Analysis Results*\n"
        header += f"_Generated at {timestamp}_\n\n"
        
        if success_count > 0:
            header += f"✅ {success_count} successful analysis"
            header += "es" if success_count > 1 else ""
        
        if failed_count > 0:
            if success_count > 0:
                header += f" | "
            header += f"❌ {failed_count} failed"
        
        header += "\n\n"
        return header
    
    def _format_successful_analysis(self, result: Dict[str, Any]) -> str:
        """Format a successful analysis result"""
        
        provider = result.get("provider", "Unknown")
        model = result.get("model", "")
        analysis = result.get("analysis", "No analysis provided")
        
        # Format provider header
        formatted = f"✅ *{provider}*"
        if model:
            formatted += f" (`{model}`)"
        formatted += "\n\n"
        
        # Process and format the analysis
        analysis = self._process_analysis_text(analysis)
        
        # Truncate if too long
        if len(analysis) > self.max_length:
            analysis = analysis[:self.max_length - 100]
            analysis += "\n\n_... (truncated for Slack - see thread for full analysis)_"
        
        formatted += analysis
        
        # Add metadata if available
        metadata = result.get("metadata", {})
        if metadata:
            formatted += self._format_metadata(metadata)
        
        return formatted
    
    def _format_failed_analysis(self, result: Dict[str, Any]) -> str:
        """Format a failed analysis result"""
        
        provider = result.get("provider", "Unknown")
        error = result.get("error", "Unknown error")
        
        return f"• *{provider}*: `{error}`\n"
    
    def _process_analysis_text(self, text: str) -> str:
        """
        Process analysis text for better Slack formatting
        
        Args:
            text: Raw analysis text
            
        Returns:
            Processed text for Slack
        """
        
        # Convert markdown headers to Slack bold
        text = re.sub(r'^#{1,3}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
        
        # Convert SQL code blocks to Slack code blocks
        text = re.sub(r'```sql\n(.*?)```', r'```\1```', text, flags=re.DOTALL)
        
        # Convert inline code to Slack code
        text = re.sub(r'`([^`]+)`', r'`\1`', text)
        
        # Convert bullet points
        text = re.sub(r'^\s*[-•]\s+', '• ', text, flags=re.MULTILINE)
        
        # Ensure proper spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata information"""
        
        formatted = "\n\n_"
        
        # Token usage for OpenAI
        if "usage" in metadata:
            usage = metadata["usage"]
            if "total_tokens" in usage:
                formatted += f"Tokens: {usage.get('total_tokens', 0)} | "
        
        # Finish reason
        if "finish_reason" in metadata:
            formatted += f"Status: {metadata['finish_reason']} | "
        
        # Remove trailing separator
        if formatted.endswith(" | "):
            formatted = formatted[:-3]
        
        formatted += "_"
        
        return formatted if len(formatted) > 5 else ""
    
    def _create_footer(self, results: List[Dict[str, Any]]) -> str:
        """Create footer with summary information"""
        
        footer = "\n" + "─" * 40 + "\n"
        footer += "💡 *Quick Actions:*\n"
        footer += "• Review and test recommended indexes in a dev environment first\n"
        footer += "• Run `EXPLAIN ANALYZE` after implementing changes\n"
        footer += "• Monitor query performance over time\n"
        
        return footer
    
    def format_sql_statements(self, statements: List[str]) -> str:
        """
        Format extracted SQL statements for easy copying
        
        Args:
            statements: List of SQL statements
            
        Returns:
            Formatted SQL block for Slack
        """
        
        if not statements:
            return "No SQL statements found"
        
        formatted = "📝 *Extracted SQL Statements:*\n```\n"
        
        for stmt in statements:
            formatted += stmt + "\n\n"
        
        formatted += "```"
        
        return formatted
    
    def format_error_message(self, error: str, context: str = None) -> str:
        """
        Format error message for Slack
        
        Args:
            error: Error message
            context: Additional context
            
        Returns:
            Formatted error message
        """
        
        formatted = "❌ *Error Occurred*\n\n"
        formatted += f"Error: `{error}`\n"
        
        if context:
            formatted += f"Context: {context}\n"
        
        formatted += "\nPlease check the logs for more details."
        
        return formatted
    
    def format_acknowledgment(self, provider_count: int) -> str:
        """
        Format initial acknowledgment message
        
        Args:
            provider_count: Number of providers being used
            
        Returns:
            Acknowledgment message
        """
        
        providers_text = "provider" if provider_count == 1 else "providers"
        
        return (
            f"🔍 *Analyzing query with {provider_count} LLM {providers_text}...*\n\n"
            f"This may take up to {config.ANALYSIS_TIMEOUT} seconds. "
            f"Results will appear in this thread."
        )