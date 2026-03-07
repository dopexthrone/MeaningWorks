"""RetrievalAgent — corpus FTS5 search + pattern extraction.

Reads:  state.intent, state.domain
Writes: state.retrieval_context

No LLM calls. Wraps persistence/sqlite_corpus.py and persistence/corpus_analysis.py.
Output includes "relevant_documents" key for CompileAgent._enrich_description().
"""

import logging
from typing import Any, Dict, List

from swarm.agents.base import SwarmAgent
from swarm.state import SwarmState

logger = logging.getLogger("motherlabs.swarm.retrieval")


class RetrievalAgent(SwarmAgent):
    """Corpus retrieval via FTS5 search and structural pattern extraction."""

    name = "retrieval"
    criticality = "medium"

    @property
    def input_keys(self) -> List[str]:
        return ["intent", "domain"]

    @property
    def output_keys(self) -> List[str]:
        return ["retrieval_context"]

    def execute(self, state: SwarmState, config: Dict[str, Any]) -> SwarmState:
        """Search corpus and extract domain patterns.

        Steps:
        1. FTS5 search on intent
        2. Domain suggestions (frequency-based components)
        3. Structural pattern extraction via CorpusAnalyzer
        4. Format results for CompileAgent consumption
        """
        from persistence.sqlite_corpus import SQLiteCorpus
        from persistence.corpus_analysis import CorpusAnalyzer

        corpus = SQLiteCorpus()

        # 1. FTS5 search
        try:
            search_results = corpus.search(state.intent)
        except Exception as e:
            logger.warning("Corpus search failed: %s", e)
            search_results = []

        # 2. Domain suggestions
        try:
            domain_suggestions = corpus.get_domain_suggestions(state.domain)
        except Exception as e:
            logger.warning("Domain suggestions failed: %s", e)
            domain_suggestions = {}

        # 3. Structural patterns via CorpusAnalyzer
        formatted_patterns = None
        try:
            analyzer = CorpusAnalyzer(corpus)
            domain_model = analyzer.build_domain_model(state.domain)
            if domain_model is not None:
                formatted_patterns = analyzer.format_corpus_patterns_section(domain_model)
        except Exception as e:
            logger.warning("Pattern extraction failed: %s", e)

        # 4. Build retrieval context
        doc_parts = []

        if search_results:
            doc_parts.append(f"Found {len(search_results)} relevant compilations.")
            for rec in search_results[:5]:
                intent = getattr(rec, "intent", getattr(rec, "description", str(rec)))
                doc_parts.append(f"- {intent}")

        if domain_suggestions and domain_suggestions.get("has_suggestions"):
            suggested = domain_suggestions.get("suggested_components", [])
            if suggested:
                doc_parts.append(f"\nDomain '{state.domain}' common components: {', '.join(str(c) for c in suggested[:10])}")

        if formatted_patterns:
            doc_parts.append(f"\n{formatted_patterns}")

        relevant_documents = "\n".join(doc_parts) if doc_parts else ""

        retrieval_context = {
            "relevant_documents": relevant_documents,
            "search_result_count": len(search_results),
            "domain_suggestions": domain_suggestions,
            "has_patterns": formatted_patterns is not None,
        }

        logger.info(
            "RetrievalAgent completed: domain=%s results=%d has_patterns=%s",
            state.domain,
            len(search_results),
            formatted_patterns is not None,
        )

        return state.with_updates(retrieval_context=retrieval_context)
