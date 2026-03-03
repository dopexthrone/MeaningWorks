"""Tests for improved semantic similarity with synonym expansion (Build 7)."""

import pytest
from kernel.closed_loop import semantic_similarity
from kernel._text_utils import (
    expand_synonyms,
    bigram_tokens,
    semantic_jaccard,
    SYNONYM_CLUSTERS,
)


class TestSynonymExpansion:
    """expand_synonyms() maps tokens to cluster representatives."""

    def test_login_expands_to_auth(self):
        tokens = {"login"}
        expanded = expand_synonyms(tokens)
        # "login" should expand to include "auth" (the cluster representative)
        assert "auth" in expanded
        assert len(expanded) > len(tokens)

    def test_unknown_token_passes_through(self):
        tokens = {"xyzzy"}
        expanded = expand_synonyms(tokens)
        assert "xyzzy" in expanded

    def test_clusters_exist(self):
        """SYNONYM_CLUSTERS has reasonable coverage."""
        assert len(SYNONYM_CLUSTERS) >= 20

    def test_bidirectional(self):
        """Both 'login' and 'auth' expand to shared cluster."""
        login_exp = expand_synonyms({"login"})
        auth_exp = expand_synonyms({"auth"})
        assert login_exp & auth_exp  # non-empty intersection


class TestBigramTokens:
    """bigram_tokens() produces stemmed bigrams."""

    def test_basic_bigrams(self):
        result = bigram_tokens("task manager calendar integration")
        assert len(result) >= 2

    def test_single_word_no_bigrams(self):
        result = bigram_tokens("hello")
        assert len(result) == 0

    def test_dedup(self):
        result = bigram_tokens("the the the")
        # After stemming and stop word removal, may be empty or minimal
        assert isinstance(result, set)


class TestSemanticJaccard:
    """semantic_jaccard() computes synonym-aware similarity."""

    def test_identical_strings(self):
        score = semantic_jaccard("authentication service", "authentication service")
        assert score >= 0.8

    def test_exact_overlap(self):
        """'auth service' and 'auth handler' share 'auth' token."""
        score = semantic_jaccard("auth service", "auth handler")
        assert score > 0.0

    def test_unrelated_strings(self):
        score = semantic_jaccard("authentication", "banana smoothie recipe")
        assert score < 0.3


class TestSemanticSimilarityImproved:
    """semantic_similarity() with synonym expansion + bigram overlap."""

    def test_synonym_aware(self):
        """'Authentication system' and 'login handler' should have non-zero similarity."""
        score = semantic_similarity(
            "Build an authentication system with user login",
            "The login handler manages user credential verification"
        )
        assert score > 0.0

    def test_positive_similarity_for_paraphrase(self):
        """Paraphrased text should have positive similarity."""
        score = semantic_similarity(
            "Build a task management application with projects and tasks",
            "The system manages tasks organized into projects for tracking"
        )
        assert score >= 0.2

    def test_recall_weighted(self):
        """Similarity is recall-weighted — reconstruction should cover original."""
        score = semantic_similarity(
            "Build auth with OAuth and JWT tokens",
            "Authentication uses OAuth for external login and JWT tokens for session management"
        )
        assert score >= 0.4

    def test_zero_for_empty(self):
        score = semantic_similarity("", "some text")
        assert score == 0.0

    def test_backward_compat(self):
        """Original token overlap still works for exact matches."""
        score = semantic_similarity(
            "Build a REST API with endpoints",
            "REST API endpoints for the application"
        )
        assert score > 0.0
