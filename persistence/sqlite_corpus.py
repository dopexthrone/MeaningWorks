"""
Motherlabs SQLite Corpus - SQLite-backed storage and indexing for compilations.

Derived from: persistence/corpus.py (JSON-backed Corpus)

Drop-in replacement for Corpus with identical public API, backed by sqlite3
instead of JSON files. Provides ACID writes, FTS5 full-text search, and
pagination support while maintaining backward-compatible JSON file output.

Tables:
  - compilations: metadata for each compilation record
  - blueprints: JSON blob storage for blueprint data
  - context_graphs: JSON blob storage for context graph data
  - compilations_fts: FTS5 virtual table for full-text search on input_text

Design decisions:
  - stdlib sqlite3 only (no external dependencies)
  - Database stored at {corpus_path}/corpus.db
  - Individual JSON files still written for backward compat
  - All writes wrapped in transactions (ACID)
  - Pagination via LIMIT/OFFSET on list methods
"""

import json
import os
import hashlib
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from persistence.corpus import CompilationRecord


class SQLiteCorpus:
    """
    SQLite-backed corpus storage for Motherlabs compilations.

    Drop-in replacement for Corpus with identical public API.
    Uses sqlite3 for metadata indexing and FTS5 for full-text search.
    Maintains backward-compatible JSON file output alongside the database.

    Storage structure:
    {corpus_path}/
    +-- corpus.db              # SQLite database
    +-- {id}/
    |   +-- context-graph.json # Backward-compat JSON files
    |   +-- blueprint.json
    |   +-- trace.md
    """

    @staticmethod
    def _default_path() -> Path:
        data_dir = os.environ.get("MOTHERLABS_DATA_DIR")
        if data_dir:
            return Path(data_dir) / "corpus"
        return Path.home() / "motherlabs" / "corpus"

    DEFAULT_PATH = Path.home() / "motherlabs" / "corpus"  # class-level fallback

    def __init__(self, corpus_path: Optional[Path] = None):
        """
        Initialize SQLite corpus.

        Args:
            corpus_path: Custom path for corpus storage (default: $MOTHERLABS_DATA_DIR/corpus or ~/motherlabs/corpus/)
        """
        self.path = corpus_path or self._default_path()
        self.db_path = self.path / "corpus.db"
        self._ensure_corpus_exists()
        self._init_db()

    def _ensure_corpus_exists(self):
        """Create corpus directory if it doesn't exist."""
        self.path.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a new database connection with standard settings.

        Returns a connection with WAL journal mode for better concurrent
        read performance, and row_factory set to sqlite3.Row for dict-like
        access to query results.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Initialize database schema if not present."""
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS compilations (
                    id TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT 'unknown',
                    timestamp TEXT NOT NULL,
                    components_count INTEGER NOT NULL DEFAULT 0,
                    insights_count INTEGER NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 0,
                    file_path TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'unknown',
                    model TEXT NOT NULL DEFAULT 'unknown',
                    stage_timings TEXT,
                    retry_counts TEXT
                );

                CREATE TABLE IF NOT EXISTS blueprints (
                    compilation_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    FOREIGN KEY (compilation_id) REFERENCES compilations(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS context_graphs (
                    compilation_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    FOREIGN KEY (compilation_id) REFERENCES compilations(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_compilations_domain
                    ON compilations(domain);

                CREATE INDEX IF NOT EXISTS idx_compilations_timestamp
                    ON compilations(timestamp);

                CREATE INDEX IF NOT EXISTS idx_compilations_provider
                    ON compilations(provider);

                CREATE INDEX IF NOT EXISTS idx_compilations_success
                    ON compilations(success);
            """)

            # Phase 12.2a: Process telemetry columns (migration for existing DBs)
            _TELEMETRY_COLUMNS = [
                ("dialogue_turns", "INTEGER"),
                ("confidence_trajectory", "TEXT"),
                ("message_type_counts", "TEXT"),
                ("conflict_count", "INTEGER"),
                ("structural_conflicts_resolved", "INTEGER"),
                ("unknown_count", "INTEGER"),
                ("dialogue_depth_config", "TEXT"),
                ("verification_score", "REAL"),
            ]
            for col_name, col_type in _TELEMETRY_COLUMNS:
                try:
                    conn.execute(
                        f"ALTER TABLE compilations ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # L2 feedback loop: corpus_feedback column (migration for existing DBs)
            try:
                conn.execute(
                    "ALTER TABLE compilations ADD COLUMN corpus_feedback TEXT"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists

            # FTS5 virtual table for full-text search.
            # Uses a standalone FTS table (no content= sync) so we manage
            # insert/delete ourselves, but avoid content-sync pitfalls.
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS compilations_fts
                    USING fts5(
                        compilation_id UNINDEXED,
                        input_text,
                        domain
                    )
                """)
            except sqlite3.OperationalError:
                # FTS table already exists or FTS5 not available
                pass

            conn.commit()
        finally:
            conn.close()

    def _generate_id(self, input_text: str) -> str:
        """Generate unique ID from input text."""
        return hashlib.sha256(input_text.encode()).hexdigest()[:12]

    def _row_to_record(self, row: sqlite3.Row) -> CompilationRecord:
        """Convert a database row to a CompilationRecord."""
        # Phase 12.2a: Safely read telemetry columns (may not exist in old DBs)
        row_keys = row.keys()

        def _get(key, deserialize_json=False):
            if key not in row_keys:
                return None
            val = row[key]
            if val is None:
                return None
            if deserialize_json:
                return json.loads(val)
            return val

        return CompilationRecord(
            id=row["id"],
            input_text=row["input_text"],
            domain=row["domain"],
            timestamp=row["timestamp"],
            components_count=row["components_count"],
            insights_count=row["insights_count"],
            success=bool(row["success"]),
            file_path=row["file_path"],
            provider=row["provider"],
            model=row["model"],
            stage_timings=json.loads(row["stage_timings"]) if row["stage_timings"] else None,
            retry_counts=json.loads(row["retry_counts"]) if row["retry_counts"] else None,
            dialogue_turns=_get("dialogue_turns"),
            confidence_trajectory=_get("confidence_trajectory", deserialize_json=True),
            message_type_counts=_get("message_type_counts", deserialize_json=True),
            conflict_count=_get("conflict_count"),
            structural_conflicts_resolved=_get("structural_conflicts_resolved"),
            unknown_count=_get("unknown_count"),
            dialogue_depth_config=_get("dialogue_depth_config", deserialize_json=True),
            verification_score=_get("verification_score"),
        )

    def _write_json_files(
        self,
        compilation_id: str,
        input_text: str,
        context_graph: Dict[str, Any],
        blueprint: Dict[str, Any],
        insights: List[str],
    ) -> Path:
        """
        Write backward-compatible JSON files alongside the database.

        Returns the compilation directory path.
        """
        compilation_dir = self.path / compilation_id
        compilation_dir.mkdir(exist_ok=True)

        context_path = compilation_dir / "context-graph.json"
        blueprint_path = compilation_dir / "blueprint.json"
        trace_path = compilation_dir / "trace.md"

        with open(context_path, "w") as f:
            json.dump(context_graph, f, indent=2)

        with open(blueprint_path, "w") as f:
            json.dump(blueprint, f, indent=2)

        with open(trace_path, "w") as f:
            f.write(f"# Compilation Trace\n\n")
            f.write(f"*ID: {compilation_id}*\n")
            f.write(f"*Generated: {datetime.now().isoformat()}*\n\n")
            f.write(f"## Input\n\n{input_text}\n\n")
            f.write(f"## Insights\n\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")

        return compilation_dir

    def store(
        self,
        input_text: str,
        context_graph: Dict[str, Any],
        blueprint: Dict[str, Any],
        insights: List[str],
        success: bool,
        provider: str = "unknown",
        model: str = "unknown",
        stage_timings: Optional[Dict[str, float]] = None,
        retry_counts: Optional[Dict[str, int]] = None,
        # Phase 12.2a: Process telemetry
        dialogue_turns: Optional[int] = None,
        confidence_trajectory: Optional[List[float]] = None,
        message_type_counts: Optional[Dict[str, int]] = None,
        conflict_count: Optional[int] = None,
        structural_conflicts_resolved: Optional[int] = None,
        unknown_count: Optional[int] = None,
        dialogue_depth_config: Optional[Dict[str, int]] = None,
        # Corpus quality weighting
        verification_score: Optional[float] = None,
        # L2 feedback loop
        corpus_feedback: Optional[str] = None,
    ) -> CompilationRecord:
        """
        Store a compilation in the corpus.

        All writes are executed in a single transaction for ACID guarantees.
        JSON files are also written for backward compatibility.

        Args:
            input_text: Original user input
            context_graph: Full context graph from compilation
            blueprint: Synthesized blueprint
            insights: List of extracted insights
            success: Whether verification passed
            provider: LLM provider used
            model: Model name used
            stage_timings: Time per stage in seconds
            retry_counts: Number of retries per stage

        Returns:
            CompilationRecord with storage metadata
        """
        compilation_id = self._generate_id(input_text)

        # Extract domain from context graph
        intent = context_graph.get("known", {}).get("intent", {})
        domain = intent.get("domain", "unknown")

        # Write JSON files for backward compat
        compilation_dir = self._write_json_files(
            compilation_id, input_text, context_graph, blueprint, insights
        )

        timestamp = datetime.now().isoformat()
        components_count = len(blueprint.get("components", []))
        insights_count = len(insights)

        # Serialize optional dicts to JSON strings for storage
        stage_timings_json = json.dumps(stage_timings) if stage_timings else None
        retry_counts_json = json.dumps(retry_counts) if retry_counts else None
        # Phase 12.2a: Serialize telemetry JSON fields
        confidence_trajectory_json = json.dumps(confidence_trajectory) if confidence_trajectory is not None else None
        message_type_counts_json = json.dumps(message_type_counts) if message_type_counts is not None else None
        dialogue_depth_config_json = json.dumps(dialogue_depth_config) if dialogue_depth_config is not None else None

        conn = self._get_connection()
        try:
            with conn:
                # Upsert compilation record
                conn.execute(
                    """
                    INSERT INTO compilations
                        (id, input_text, domain, timestamp, components_count,
                         insights_count, success, file_path, provider, model,
                         stage_timings, retry_counts,
                         dialogue_turns, confidence_trajectory, message_type_counts,
                         conflict_count, structural_conflicts_resolved,
                         unknown_count, dialogue_depth_config, verification_score,
                         corpus_feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        input_text=excluded.input_text,
                        domain=excluded.domain,
                        timestamp=excluded.timestamp,
                        components_count=excluded.components_count,
                        insights_count=excluded.insights_count,
                        success=excluded.success,
                        file_path=excluded.file_path,
                        provider=excluded.provider,
                        model=excluded.model,
                        stage_timings=excluded.stage_timings,
                        retry_counts=excluded.retry_counts,
                        dialogue_turns=excluded.dialogue_turns,
                        confidence_trajectory=excluded.confidence_trajectory,
                        message_type_counts=excluded.message_type_counts,
                        conflict_count=excluded.conflict_count,
                        structural_conflicts_resolved=excluded.structural_conflicts_resolved,
                        unknown_count=excluded.unknown_count,
                        dialogue_depth_config=excluded.dialogue_depth_config,
                        verification_score=excluded.verification_score,
                        corpus_feedback=excluded.corpus_feedback
                    """,
                    (
                        compilation_id,
                        input_text[:500],  # Truncate for index, matching Corpus behavior
                        domain,
                        timestamp,
                        components_count,
                        insights_count,
                        int(success),
                        str(compilation_dir),
                        provider,
                        model,
                        stage_timings_json,
                        retry_counts_json,
                        dialogue_turns,
                        confidence_trajectory_json,
                        message_type_counts_json,
                        conflict_count,
                        structural_conflicts_resolved,
                        unknown_count,
                        dialogue_depth_config_json,
                        verification_score,
                        corpus_feedback,
                    ),
                )

                # Upsert blueprint blob
                conn.execute(
                    """
                    INSERT INTO blueprints (compilation_id, data)
                    VALUES (?, ?)
                    ON CONFLICT(compilation_id) DO UPDATE SET data=excluded.data
                    """,
                    (compilation_id, json.dumps(blueprint)),
                )

                # Upsert context graph blob
                conn.execute(
                    """
                    INSERT INTO context_graphs (compilation_id, data)
                    VALUES (?, ?)
                    ON CONFLICT(compilation_id) DO UPDATE SET data=excluded.data
                    """,
                    (compilation_id, json.dumps(context_graph)),
                )

                # Update FTS index
                # Delete old entry if exists, then insert new
                conn.execute(
                    "DELETE FROM compilations_fts WHERE compilation_id = ?",
                    (compilation_id,),
                )
                conn.execute(
                    """
                    INSERT INTO compilations_fts (compilation_id, input_text, domain)
                    VALUES (?, ?, ?)
                    """,
                    (compilation_id, input_text[:500], domain),
                )
        finally:
            conn.close()

        return CompilationRecord(
            id=compilation_id,
            input_text=input_text[:500],
            domain=domain,
            timestamp=timestamp,
            components_count=components_count,
            insights_count=insights_count,
            success=success,
            file_path=str(compilation_dir),
            provider=provider,
            model=model,
            stage_timings=stage_timings,
            retry_counts=retry_counts,
            dialogue_turns=dialogue_turns,
            confidence_trajectory=confidence_trajectory,
            message_type_counts=message_type_counts,
            conflict_count=conflict_count,
            structural_conflicts_resolved=structural_conflicts_resolved,
            unknown_count=unknown_count,
            dialogue_depth_config=dialogue_depth_config,
            verification_score=verification_score,
        )

    def list_all(
        self,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> List[CompilationRecord]:
        """
        List all compilations in corpus, sorted by timestamp descending.

        Args:
            page: Page number (1-based). None returns all results.
            per_page: Results per page. None returns all results.

        Returns:
            List of CompilationRecord sorted by timestamp (newest first)
        """
        conn = self._get_connection()
        try:
            query = "SELECT * FROM compilations ORDER BY timestamp DESC"
            params: list = []

            if page is not None and per_page is not None:
                offset = (page - 1) * per_page
                query += " LIMIT ? OFFSET ?"
                params.extend([per_page, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def list_by_domain(
        self,
        domain: str,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> List[CompilationRecord]:
        """
        List compilations filtered by domain.

        Args:
            domain: Domain name to filter by (case-insensitive)
            page: Page number (1-based). None returns all results.
            per_page: Results per page. None returns all results.

        Returns:
            List of CompilationRecord in the given domain, newest first
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT * FROM compilations
                WHERE LOWER(domain) = LOWER(?)
                ORDER BY timestamp DESC
            """
            params: list = [domain]

            if page is not None and per_page is not None:
                offset = (page - 1) * per_page
                query += " LIMIT ? OFFSET ?"
                params.extend([per_page, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def get(self, compilation_id: str) -> Optional[CompilationRecord]:
        """Get compilation record by ID."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM compilations WHERE id = ?",
                (compilation_id,),
            ).fetchone()
            if row:
                return self._row_to_record(row)
            return None
        finally:
            conn.close()

    def load_context_graph(self, compilation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load full context graph for a compilation.

        Reads from SQLite first, falls back to JSON file for backward compat.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT data FROM context_graphs WHERE compilation_id = ?",
                (compilation_id,),
            ).fetchone()
            if row:
                return json.loads(row["data"])
        finally:
            conn.close()

        # Fallback: try reading from JSON file
        record = self.get(compilation_id)
        if record:
            context_path = Path(record.file_path) / "context-graph.json"
            if context_path.exists():
                with open(context_path, "r") as f:
                    return json.load(f)
        return None

    def load_blueprint(self, compilation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load blueprint for a compilation.

        Reads from SQLite first, falls back to JSON file for backward compat.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT data FROM blueprints WHERE compilation_id = ?",
                (compilation_id,),
            ).fetchone()
            if row:
                return json.loads(row["data"])
        finally:
            conn.close()

        # Fallback: try reading from JSON file
        record = self.get(compilation_id)
        if record:
            blueprint_path = Path(record.file_path) / "blueprint.json"
            if blueprint_path.exists():
                with open(blueprint_path, "r") as f:
                    return json.load(f)
        return None

    def search(self, query: str) -> List[CompilationRecord]:
        """
        Search compilations using FTS5 full-text search.

        Falls back to LIKE-based substring search if FTS fails.

        Args:
            query: Search string

        Returns:
            Matching compilation records sorted by relevance
        """
        conn = self._get_connection()
        try:
            # Try FTS5 search first
            try:
                # Escape FTS5 special characters and build match expression
                safe_query = query.replace('"', '""')
                rows = conn.execute(
                    """
                    SELECT c.* FROM compilations c
                    JOIN compilations_fts fts ON c.id = fts.compilation_id
                    WHERE compilations_fts MATCH ?
                    ORDER BY fts.rank
                    """,
                    (f'"{safe_query}"',),
                ).fetchall()
                return [self._row_to_record(row) for row in rows]
            except sqlite3.OperationalError:
                pass

            # Fallback: LIKE-based substring search
            query_pattern = f"%{query}%"
            rows = conn.execute(
                """
                SELECT * FROM compilations
                WHERE LOWER(input_text) LIKE LOWER(?)
                   OR LOWER(domain) LIKE LOWER(?)
                ORDER BY timestamp DESC
                """,
                (query_pattern, query_pattern),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics including suggestion readiness."""
        conn = self._get_connection()
        try:
            # Total compilations
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM compilations"
            ).fetchone()["cnt"]

            if total == 0:
                return {
                    "total_compilations": 0,
                    "domains": {},
                    "success_rate": 0.0,
                    "total_components": 0,
                    "total_insights": 0,
                    "domains_with_suggestions": [],
                }

            # Domain counts
            domain_rows = conn.execute(
                "SELECT domain, COUNT(*) as cnt FROM compilations GROUP BY domain"
            ).fetchall()
            domains = {row["domain"]: row["cnt"] for row in domain_rows}

            # Success rate
            successful = conn.execute(
                "SELECT COUNT(*) as cnt FROM compilations WHERE success = 1"
            ).fetchone()["cnt"]

            # Totals
            totals = conn.execute(
                """
                SELECT
                    COALESCE(SUM(components_count), 0) as total_components,
                    COALESCE(SUM(insights_count), 0) as total_insights
                FROM compilations
                """
            ).fetchone()

            # Domains with enough successful samples for suggestions
            min_samples = 3
            suggestion_rows = conn.execute(
                """
                SELECT domain, COUNT(*) as cnt
                FROM compilations
                WHERE success = 1
                GROUP BY domain
                HAVING cnt >= ?
                """,
                (min_samples,),
            ).fetchall()
            domains_with_suggestions = [row["domain"] for row in suggestion_rows]

            return {
                "total_compilations": total,
                "domains": domains,
                "success_rate": successful / total if total else 0.0,
                "total_components": totals["total_components"],
                "total_insights": totals["total_insights"],
                "domains_with_suggestions": domains_with_suggestions,
            }
        finally:
            conn.close()

    def get_provider_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics by provider.

        Args:
            provider: Optional filter for specific provider

        Returns:
            Dict mapping provider name to stats
        """
        conn = self._get_connection()
        try:
            if provider:
                rows = conn.execute(
                    "SELECT * FROM compilations WHERE provider = ?",
                    (provider,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM compilations").fetchall()

            records = [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

        # Group records by provider
        provider_records: Dict[str, List[CompilationRecord]] = {}
        for r in records:
            p = r.provider
            if p not in provider_records:
                provider_records[p] = []
            provider_records[p].append(r)

        result: Dict[str, Any] = {}
        for p, p_records in provider_records.items():
            successful = sum(1 for r in p_records if r.success)
            models = list(set(r.model for r in p_records))

            # Calculate average synthesis time if available
            synthesis_times = [
                r.stage_timings.get("synthesis", 0)
                for r in p_records
                if r.stage_timings
            ]
            avg_synthesis = (
                sum(synthesis_times) / len(synthesis_times)
                if synthesis_times
                else 0.0
            )

            # Calculate average retries if available
            retry_list = [
                r.retry_counts.get("synthesis", 0)
                for r in p_records
                if r.retry_counts
            ]
            avg_retries = (
                sum(retry_list) / len(retry_list) if retry_list else 0.0
            )

            result[p] = {
                "total_compilations": len(p_records),
                "success_rate": successful / len(p_records) if p_records else 0.0,
                "avg_synthesis_time": round(avg_synthesis, 2),
                "avg_retries": round(avg_retries, 2),
                "models_used": models,
            }

        return result

    def export_for_recompile(self, compilation_id: str) -> Optional[str]:
        """
        Export a prior compilation as input for re-compilation.

        Returns:
            Formatted string suitable as input to a new compile() call
        """
        context = self.load_context_graph(compilation_id)
        if not context:
            return None

        record = self.get(compilation_id)
        if not record:
            return None

        # Build re-compile prompt from prior context
        insights = context.get("insights", [])
        known = context.get("known", {})
        intent = known.get("intent", {})

        output = f"""Prior compilation: {record.id}
Domain: {record.domain}

Original input: {record.input_text}

Core need identified: {intent.get('core_need', 'unknown')}

Key insights from prior compilation:
"""
        for i, insight in enumerate(insights[:10], 1):  # Top 10 insights
            output += f"  {i}. {insight}\n"

        output += f"""
Components discovered: {record.components_count}

Build on this prior understanding. Refine, extend, or challenge these insights.
"""
        return output

    # =========================================================================
    # CORPUS-DRIVEN SUGGESTIONS
    # =========================================================================

    def _analyze_component_frequencies(
        self,
        records: List[CompilationRecord],
    ) -> Dict[str, float]:
        """
        Analyze component frequencies across compilations.

        Args:
            records: List of compilation records to analyze

        Returns:
            Dict mapping component name -> frequency (0.0 to 1.0)
        """
        if not records:
            return {}

        component_counts: Dict[str, int] = {}
        total = len(records)

        for record in records:
            blueprint = self.load_blueprint(record.id)
            if not blueprint:
                continue

            seen_in_blueprint: set = set()
            for comp in blueprint.get("components", []):
                name = comp.get("name")
                if name and name not in seen_in_blueprint:
                    seen_in_blueprint.add(name)
                    component_counts[name] = component_counts.get(name, 0) + 1

        return {name: count / total for name, count in component_counts.items()}

    def _analyze_relationship_frequencies(
        self,
        records: List[CompilationRecord],
    ) -> Dict[tuple, float]:
        """
        Analyze relationship frequencies across compilations.

        Args:
            records: List of compilation records to analyze

        Returns:
            Dict mapping (from, to, type) tuple -> frequency (0.0 to 1.0)
        """
        if not records:
            return {}

        rel_counts: Dict[tuple, int] = {}
        total = len(records)

        for record in records:
            blueprint = self.load_blueprint(record.id)
            if not blueprint:
                continue

            seen_in_blueprint: set = set()
            for rel in blueprint.get("relationships", []):
                key = (rel.get("from"), rel.get("to"), rel.get("type"))
                if key not in seen_in_blueprint and all(key):
                    seen_in_blueprint.add(key)
                    rel_counts[key] = rel_counts.get(key, 0) + 1

        return {rel: count / total for rel, count in rel_counts.items()}

    def get_domain_suggestions(
        self,
        domain: str,
        min_frequency: float = 0.6,
        min_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Get suggested canonical components for a domain based on corpus patterns.

        Args:
            domain: Target domain to find suggestions for
            min_frequency: Minimum frequency threshold (default 0.6 = 60%)
            min_samples: Minimum successful compilations required (default 3)

        Returns:
            Dict with domain, sample_size, suggested_components,
            suggested_relationships, component_frequencies, has_suggestions
        """
        records = self.list_by_domain(domain)
        successful = [r for r in records if r.success]

        if len(successful) < min_samples:
            return {
                "domain": domain,
                "sample_size": len(successful),
                "suggested_components": [],
                "suggested_relationships": [],
                "component_frequencies": {},
                "has_suggestions": False,
                "reason": f"Insufficient samples ({len(successful)}/{min_samples})",
            }

        comp_frequencies = self._analyze_component_frequencies(successful)

        suggested_components = [
            name
            for name, freq in comp_frequencies.items()
            if freq >= min_frequency
        ]
        suggested_components.sort(
            key=lambda n: comp_frequencies.get(n, 0), reverse=True
        )

        rel_frequencies = self._analyze_relationship_frequencies(successful)
        rel_threshold = min_frequency * 0.67
        suggested_relationships = [
            rel for rel, freq in rel_frequencies.items() if freq >= rel_threshold
        ]

        return {
            "domain": domain,
            "sample_size": len(successful),
            "suggested_components": suggested_components,
            "suggested_relationships": suggested_relationships,
            "component_frequencies": comp_frequencies,
            "has_suggestions": len(suggested_components) > 0,
        }

    # =========================================================================
    # L2 FEEDBACK LOOP
    # =========================================================================

    def get_adoption_data(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get corpus feedback and verification scores for a domain.

        Returns list of {"feedback": parsed_json, "verification_score": float}
        for all compilations with stored corpus_feedback in the domain.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT corpus_feedback, verification_score
                FROM compilations
                WHERE LOWER(domain) = LOWER(?)
                  AND corpus_feedback IS NOT NULL
                ORDER BY timestamp DESC
                """,
                (domain,),
            ).fetchall()
            results = []
            for row in rows:
                try:
                    feedback = json.loads(row["corpus_feedback"])
                except (json.JSONDecodeError, TypeError):
                    continue
                v_score = row["verification_score"] if "verification_score" in row.keys() else None
                results.append({
                    "feedback": feedback,
                    "verification_score": v_score if v_score is not None else 0.0,
                })
            return results
        finally:
            conn.close()

    # =========================================================================
    # SQLITE-SPECIFIC UTILITIES
    # =========================================================================

    def count(self) -> int:
        """Return total number of compilations in the corpus."""
        conn = self._get_connection()
        try:
            return conn.execute(
                "SELECT COUNT(*) as cnt FROM compilations"
            ).fetchone()["cnt"]
        finally:
            conn.close()

    def count_by_domain(self, domain: str) -> int:
        """Return number of compilations in a specific domain."""
        conn = self._get_connection()
        try:
            return conn.execute(
                "SELECT COUNT(*) as cnt FROM compilations WHERE LOWER(domain) = LOWER(?)",
                (domain,),
            ).fetchone()["cnt"]
        finally:
            conn.close()
