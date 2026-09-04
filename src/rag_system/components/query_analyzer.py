"""Query Analyzer — classifies queries, extracts metadata filters, and routes to the right pipeline.

Responsibilities:
  1. Classify query complexity → route to cheap vs. expensive model
  2. Extract entity filters (article number, doc type, company ticker, date range)
  3. Detect query intent (factual / numeric / comparative / agentic)
  4. Detect prompt injection before hitting the pipeline
  5. Rewrite ambiguous queries for better retrieval precision

Uses rule-based extraction for speed (no LLM call), with optional LLM rewrite for complex cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class QueryIntent(str, Enum):
    """Query intent classification."""

    FACTUAL = "factual"  # "What was revenue in Q3?" — simple lookup
    NUMERIC = "numeric"  # "Calculate CAGR from 2020 to 2023" — needs PoT
    COMPARATIVE = "comparative"  # "Compare gross margins across segments" — multi-chunk
    TEMPORAL = "temporal"  # "How has margin trended over 8 quarters?" — time-series
    AGENTIC = "agentic"  # "Identify anomalies and flag risks" — multi-step
    UNKNOWN = "unknown"


class QueryComplexity(str, Enum):
    """Query complexity classification."""

    SIMPLE = "simple"  # Route to gpt-4o-mini
    MODERATE = "moderate"  # Route to gpt-4o-mini with larger context
    COMPLEX = "complex"  # Route to gpt-4o


@dataclass
class QueryAnalysis:
    """Full analysis result for a single query."""

    original_query: str
    rewritten_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    is_injection: bool = False
    injection_reason: Optional[str] = None
    suggested_top_k: int = 5
    suggested_model_override: Optional[str] = None
    use_pot: bool = False
    use_agentic: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ──────────────────────────────────────────────────────────────────────────────

_STRONG_NUMERIC_RE = re.compile(r"\b(calculat|cagr|compound.annual|growth rate)\b", re.I)
_NUMERIC_RE = re.compile(
    r"\b(calculat|cagr|compound.annual|growth rate|eps|ebitda|margin|"
    r"return|ratio|percent|increase|decrease|compare|versus|yoy|qoq|basis.point|"
    r"how much|how many|sum|average|mean|median)\b",
    re.I,
)
_COMPARATIVE_RE = re.compile(
    r"\b(compare|versus|vs\.?|relative to|difference|between|across segments|"
    r"year.over.year|quarter.over.quarter|benchmark|against)\b",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(trend|trended|over time|over the.+quarter|historical|past \d+ quarter|"
    r"trajectory|evolution|since \d{4}|from \d{4} to \d{4})\b",
    re.I,
)
_AGENTIC_RE = re.compile(
    r"\b(anomal|flag|identify.+risk|find all|summarize across|extract all|"
    r"multi.step|analyze and report|comprehensive|end.to.end)\b",
    re.I,
)
_INJECTION_RE = re.compile(
    r"(ignore previous|disregard|forget your|jailbreak|act as|you are now|"
    r"new persona|system prompt|bypass|override instructions|pretend you|"
    r"roleplay as|developer mode|dan mode)",
    re.I,
)

# Financial entity extractors
_ARTICLE_RE = re.compile(r"\bArticle\s+(\w+)\b", re.I)
_SECTION_RE = re.compile(r"\bSection\s+([\d\.]+[A-Za-z]?)\b", re.I)
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b(?=\s+(?:stock|shares|equity|Inc|Corp|Ltd))")
_DOC_TYPE_RE = re.compile(
    r"\b(10-K|10-Q|8-K|proxy|DEF\s*14A|earnings\s+release|annual\s+report|"
    r"quarterly\s+report|investor\s+presentation|credit\s+agreement)\b",
    re.I,
)
_DATE_RE = re.compile(r"\b(Q[1-4]\s*\d{4}|FY\s*\d{4}|\d{4})\b")
_COMPANY_RE = re.compile(
    r"\b(Tesla|Apple|Microsoft|Google|Amazon|Meta|NVIDIA|Goldman|JPMorgan|"
    r"BlackRock|Berkshire|Netflix|Salesforce|Alphabet)\b",
    re.I,
)


class QueryAnalyzer:
    """Rule-based query analyzer with optional LLM rewrite for complex cases."""

    def __init__(self, enable_llm_rewrite: bool = False) -> None:
        """Initialize the query analyzer.

        Args:
            enable_llm_rewrite: Whether to enable LLM-based query rewriting.
        """
        self._llm_rewrite = enable_llm_rewrite

    def analyze(self, query: str, tenant_id: Optional[str] = None) -> QueryAnalysis:
        """Fully analyze a query and return structured QueryAnalysis.

        Args:
            query: The query string to analyze.
            tenant_id: Optional tenant identifier for logging.

        Returns:
            QueryAnalysis object with classification and extracted entities.
        """
        # 1. Injection check (always first)
        injection_match = _INJECTION_RE.search(query)
        if injection_match:
            logger.warning(
                "query_injection_detected",
                pattern=injection_match.group(0),
                tenant_id=tenant_id,
                query_preview=query[:100],
            )
            return QueryAnalysis(
                original_query=query,
                rewritten_query=query,
                intent=QueryIntent.UNKNOWN,
                complexity=QueryComplexity.SIMPLE,
                is_injection=True,
                injection_reason=f"Blocked pattern: '{injection_match.group(0)}'",
            )

        # 2. Intent classification
        intent = self._classify_intent(query)

        # 3. Complexity
        complexity = self._classify_complexity(query, intent)

        # 4. Entity extraction → metadata filters
        filters, entities = self._extract_entities(query)

        # 5. Query rewrite for ambiguous article/section refs
        rewritten = self._rewrite_query(query, entities)

        # 6. Derived flags
        use_pot = intent == QueryIntent.NUMERIC
        use_agentic = intent == QueryIntent.AGENTIC
        top_k = (
            10
            if intent in (QueryIntent.COMPARATIVE, QueryIntent.TEMPORAL, QueryIntent.AGENTIC)
            else 5
        )
        model_override = "gpt-4o" if complexity == QueryComplexity.COMPLEX else None

        result = QueryAnalysis(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            complexity=complexity,
            metadata_filters=filters,
            extracted_entities=entities,
            suggested_top_k=top_k,
            suggested_model_override=model_override,
            use_pot=use_pot,
            use_agentic=use_agentic,
        )
        logger.debug(
            "query_analyzed",
            intent=intent.value,
            complexity=complexity.value,
            filters=filters,
            use_pot=use_pot,
            tenant_id=tenant_id,
        )
        return result

    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a query.

        Args:
            query: The query string.

        Returns:
            QueryIntent classification.
        """
        if _AGENTIC_RE.search(query):
            return QueryIntent.AGENTIC
        if _STRONG_NUMERIC_RE.search(query):
            return QueryIntent.NUMERIC
        if _TEMPORAL_RE.search(query):
            return QueryIntent.TEMPORAL
        if _COMPARATIVE_RE.search(query):
            return QueryIntent.COMPARATIVE
        if _NUMERIC_RE.search(query):
            return QueryIntent.NUMERIC
        return QueryIntent.FACTUAL

    def _classify_complexity(self, query: str, intent: QueryIntent) -> QueryComplexity:
        """Classify the complexity of a query.

        Args:
            query: The query string.
            intent: The classified intent.

        Returns:
            QueryComplexity classification.
        """
        if intent in (QueryIntent.AGENTIC, QueryIntent.COMPARATIVE):
            return QueryComplexity.COMPLEX
        if intent in (QueryIntent.TEMPORAL, QueryIntent.NUMERIC):
            return QueryComplexity.MODERATE
        # Long queries often need more reasoning
        if len(query.split()) > 30:
            return QueryComplexity.MODERATE
        return QueryComplexity.SIMPLE

    def _extract_entities(self, query: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Extract entities from the query.

        Args:
            query: The query string.

        Returns:
            Tuple of (filters dict, entities dict).
        """
        filters: Dict[str, Any] = {}
        entities: Dict[str, List[str]] = {}

        articles = _ARTICLE_RE.findall(query)
        if articles:
            entities["article_numbers"] = articles
            filters["article_number"] = articles[0]  # primary filter

        sections = _SECTION_RE.findall(query)
        if sections:
            entities["sections"] = sections
            filters["section"] = sections[0]

        doc_types = _DOC_TYPE_RE.findall(query)
        if doc_types:
            entities["doc_types"] = [d.upper().replace(" ", "_") for d in doc_types]
            filters["doc_type"] = entities["doc_types"][0]

        dates = _DATE_RE.findall(query)
        if dates:
            entities["dates"] = dates

        companies = _COMPANY_RE.findall(query)
        if companies:
            entities["companies"] = companies

        return filters, entities

    def _rewrite_query(self, query: str, entities: Dict[str, List[str]]) -> str:
        """Add disambiguating context for article/section references.

        Args:
            query: The original query.
            entities: Extracted entities.

        Returns:
            Rewritten query with additional context.
        """
        if not entities.get("article_numbers") and not entities.get("sections"):
            return query

        additions = []
        if entities.get("article_numbers"):
            nums = ", ".join(entities["article_numbers"])
            additions.append(f"Focus specifically on Article {nums} (not other numbered articles)")
        if entities.get("sections"):
            secs = ", ".join(entities["sections"])
            additions.append(f"Extract exact content from Section {secs}")

        if additions:
            return f"{query} [{'; '.join(additions)}]"
        return query

    def should_skip_vision(self, query: str) -> bool:
        """Return True if query is clearly text-only (skip expensive vision path).

        Args:
            query: The query string.

        Returns:
            True if vision processing should be skipped.
        """
        text_only_re = re.compile(
            r"\b(who signed|what date|legal entity|counterparty|governing law|"
            r"notice period|definition of|defined term|whereas|hereby)\b",
            re.I,
        )
        return bool(text_only_re.search(query))

    def batch_analyze(
        self, queries: List[str], tenant_id: Optional[str] = None
    ) -> List[QueryAnalysis]:
        """Analyze multiple queries.

        Args:
            queries: List of query strings.
            tenant_id: Optional tenant identifier.

        Returns:
            List of QueryAnalysis results.
        """
        return [self.analyze(q, tenant_id=tenant_id) for q in queries]
