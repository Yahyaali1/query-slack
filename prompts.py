"""
prompts.py - Centralized prompt templates for LLM analysis
"""

from typing import Dict, Any


class PromptTemplates:
    """Prompt templates for PostgreSQL query analysis"""

    @staticmethod
    def get_analysis_prompt(query_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis prompt from query data

        Args:
            query_data: Dictionary containing query, explain output, and metadata

        Returns:
            Formatted prompt string for LLM analysis
        """

        # Extract data with safe defaults
        execution_time = query_data.get('execution_time', 'N/A')
        rows_returned = query_data.get('rows', 'N/A')
        database_name = query_data.get('database', 'N/A')
        table_size = query_data.get('table_size', 'N/A')
        query_text = query_data.get('query', 'No query provided')
        explain_output = query_data.get('explain_output', 'No EXPLAIN output provided')

        return f"""You are an expert PostgreSQL database performance engineer with deep knowledge of query optimization, indexing strategies, and database internals. Analyze the following slow query and provide actionable optimization recommendations.

**CONTEXT:**
- Execution Time: {execution_time}ms
- Rows Returned: {rows_returned}
- Database: {database_name}
- Table Size: {table_size}

**SQL QUERY:**
```sql
{query_text}
```

**EXPLAIN ANALYZE OUTPUT:**
```
{explain_output}
```

**ANALYSIS REQUIRED:**

1. **Root Cause Analysis** 
   • Identify the PRIMARY bottleneck causing slow performance
   • Reference specific nodes in the EXPLAIN plan (with costs/times)
   • Explain WHY this is the critical issue

2. **Performance Issues Detected**
   • List ALL performance problems found, ordered by impact:
     - Sequential scans on large tables
     - Missing or unused indexes
     - Inefficient join methods (nested loops on large sets)
     - Poor cardinality estimates
     - Excessive buffer usage
     - Sort/hash operations spilling to disk
   • Include the specific line from EXPLAIN output for each issue

3. **Optimization Recommendations**
   Priority 1 - IMMEDIATE ACTIONS:
   • Provide EXACT SQL statements that can be executed:
     - CREATE INDEX statements with column order reasoning
     - Query rewrites with explanation
     - ANALYZE/VACUUM commands if needed

   Priority 2 - QUERY REFACTORING:
   • Alternative query structure (if applicable)
   • CTE vs subquery recommendations
   • Join order optimization suggestions

4. **Index Strategy**
   • Exact CREATE INDEX statements needed:
     ```sql
     CREATE INDEX idx_name ON table(columns) WHERE conditions;
     ```
   • Explain why each index helps (which operations it optimizes)
   • Consider partial, expression, or covering indexes if beneficial
   • Mention any indexes that should be DROPPED

5. **Query Rewrite** (if beneficial)
   • Provide the COMPLETE rewritten query
   • Highlight what changed and why
   • Expected performance improvement

6. **Performance Impact Estimate**
   • Quantify expected improvement: "Reduce execution time by X%"
   • Before: Current metrics
   • After: Expected metrics
   • Reasoning for estimates

7. **Additional Considerations**
   • Table statistics updates needed? (ANALYZE)
   • Autovacuum tuning required?
   • Configuration parameters to review (work_mem, random_page_cost, etc.)
   • Data model changes for long-term improvement
   • Risks or trade-offs (index maintenance overhead, disk space)

**IMPORTANT:** 
- Be specific and actionable - provide copy-paste ready SQL
- Reference actual numbers from the EXPLAIN output
- Prioritize recommendations by impact
- Consider both quick wins and long-term solutions

Format your response with clear sections using markdown headers and code blocks for SQL statements."""

    @staticmethod
    def get_index_analysis_prompt(query_data: Dict[str, Any]) -> str:
        """
        Generate a focused prompt for index-specific analysis

        Args:
            query_data: Dictionary containing query and current indexes

        Returns:
            Formatted prompt for index recommendations
        """

        query_text = query_data.get('query', '')
        current_indexes = query_data.get('current_indexes', 'Not provided')
        explain_output = query_data.get('explain_output', '')

        return f"""As a PostgreSQL indexing expert, analyze this query and recommend optimal indexes.

**QUERY:**
```sql
{query_text}
```

**CURRENT INDEXES:**
```
{current_indexes}
```

**EXPLAIN OUTPUT:**
```
{explain_output}
```

Provide:
1. **Missing Indexes** - CREATE INDEX statements for indexes that should be added
2. **Redundant Indexes** - DROP INDEX statements for unnecessary indexes  
3. **Index Modifications** - Suggestions to improve existing indexes
4. **Reasoning** - Why each index change will improve performance
5. **Trade-offs** - Storage and write performance impact

Focus on:
- B-tree vs other index types (GiST, GIN, BRIN)
- Partial indexes for filtered queries
- Expression indexes for computed columns
- Covering indexes to enable index-only scans
- Multi-column index column ordering

Provide exact, executable SQL statements."""

    @staticmethod
    def get_comparison_prompt(analyses: list) -> str:
        """
        Generate a prompt to compare and synthesize multiple LLM analyses

        Args:
            analyses: List of analysis results from different providers

        Returns:
            Formatted prompt for comparison
        """

        analyses_text = "\n\n---\n\n".join([
            f"**{a.get('provider', 'Unknown')} Analysis:**\n{a.get('analysis', '')}"
            for a in analyses if a.get('success', False)
        ])

        return f"""Compare and synthesize these PostgreSQL query optimization analyses from different sources:

{analyses_text}

Provide a UNIFIED recommendation that:
1. **Consensus Points** - What all analyses agree on
2. **Unique Insights** - Valuable points from individual analyses
3. **Conflicts** - Where analyses disagree and your recommendation
4. **Final Action Plan** - Prioritized list of SQL statements to execute
5. **Risk Assessment** - Potential issues to watch for

Focus on actionable outcomes, not the analysis process."""