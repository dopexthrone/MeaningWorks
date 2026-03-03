"""
Motherlabs Corpus Migrations - JSON to SQLite migration utility.

Provides tooling to migrate an existing JSON-backed Corpus to the new
SQLite-backed SQLiteCorpus format. Handles empty/missing data gracefully.

Usage:
    from persistence.migrations import migrate_json_to_sqlite
    migrate_json_to_sqlite(Path("~/motherlabs/corpus"), Path("~/motherlabs/corpus"))
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from persistence.corpus import Corpus, CompilationRecord
from persistence.sqlite_corpus import SQLiteCorpus

logger = logging.getLogger(__name__)


class MigrationResult:
    """Result summary from a JSON-to-SQLite migration."""

    def __init__(self):
        self.records_found: int = 0
        self.records_migrated: int = 0
        self.records_skipped: int = 0
        self.records_failed: int = 0
        self.errors: List[str] = []

    @property
    def success(self) -> bool:
        return self.records_failed == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records_found": self.records_found,
            "records_migrated": self.records_migrated,
            "records_skipped": self.records_skipped,
            "records_failed": self.records_failed,
            "success": self.success,
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        return (
            f"MigrationResult(found={self.records_found}, "
            f"migrated={self.records_migrated}, "
            f"skipped={self.records_skipped}, "
            f"failed={self.records_failed})"
        )


def migrate_json_to_sqlite(
    json_corpus_path: Path,
    sqlite_corpus_path: Optional[Path] = None,
) -> MigrationResult:
    """
    Migrate an existing JSON-backed corpus to SQLite.

    Reads the JSON index and all associated files (context graphs, blueprints),
    then populates the SQLite database. Existing SQLite entries with the same
    ID are overwritten (upsert).

    Args:
        json_corpus_path: Path to existing JSON corpus directory
            (containing index.json and compilation subdirectories)
        sqlite_corpus_path: Path for SQLite corpus directory.
            If None, uses the same path as json_corpus_path
            (the SQLite DB is created alongside the JSON files).

    Returns:
        MigrationResult with counts and any errors encountered
    """
    result = MigrationResult()

    # Use same path if not specified
    if sqlite_corpus_path is None:
        sqlite_corpus_path = json_corpus_path

    json_corpus_path = Path(json_corpus_path)
    sqlite_corpus_path = Path(sqlite_corpus_path)

    # Check if JSON corpus exists
    index_path = json_corpus_path / "index.json"
    if not index_path.exists():
        logger.info("No JSON index found at %s - nothing to migrate", index_path)
        return result

    # Load JSON index
    try:
        with open(index_path, "r") as f:
            raw_records = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        result.errors.append(f"Failed to read index.json: {e}")
        return result

    if not isinstance(raw_records, list):
        result.errors.append("index.json does not contain a list")
        return result

    result.records_found = len(raw_records)

    if result.records_found == 0:
        logger.info("JSON index is empty - nothing to migrate")
        return result

    # Parse records
    records: List[CompilationRecord] = []
    for raw in raw_records:
        try:
            records.append(CompilationRecord.from_dict(raw))
        except Exception as e:
            result.records_failed += 1
            result.errors.append(f"Failed to parse record: {e}")

    # Initialize SQLite corpus
    sqlite_corpus = SQLiteCorpus(corpus_path=sqlite_corpus_path)

    # Migrate each record
    for record in records:
        try:
            # Load context graph from JSON file
            context_graph = _load_json_file(
                json_corpus_path / record.id / "context-graph.json"
            )
            if context_graph is None:
                # Try using file_path from record
                context_graph = _load_json_file(
                    Path(record.file_path) / "context-graph.json"
                )

            # Load blueprint from JSON file
            blueprint = _load_json_file(
                json_corpus_path / record.id / "blueprint.json"
            )
            if blueprint is None:
                blueprint = _load_json_file(
                    Path(record.file_path) / "blueprint.json"
                )

            if context_graph is None or blueprint is None:
                result.records_skipped += 1
                logger.warning(
                    "Skipping record %s: missing context graph or blueprint",
                    record.id,
                )
                continue

            # Extract insights from context graph
            insights = context_graph.get("insights", [])

            # Re-store into SQLite corpus
            # We need to reconstruct the full input_text and context_graph
            # to pass through the standard store() method. However, store()
            # regenerates the ID from input_text and re-extracts domain,
            # so the result should be identical.
            sqlite_corpus.store(
                input_text=record.input_text,
                context_graph=context_graph,
                blueprint=blueprint,
                insights=insights,
                success=record.success,
                provider=record.provider,
                model=record.model,
                stage_timings=record.stage_timings,
                retry_counts=record.retry_counts,
            )

            result.records_migrated += 1
            logger.debug("Migrated record %s", record.id)

        except Exception as e:
            result.records_failed += 1
            result.errors.append(f"Failed to migrate record {record.id}: {e}")
            logger.error("Failed to migrate record %s: %s", record.id, e)

    logger.info(
        "Migration complete: %d found, %d migrated, %d skipped, %d failed",
        result.records_found,
        result.records_migrated,
        result.records_skipped,
        result.records_failed,
    )

    return result


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON file, returning None on any error.

    Args:
        path: Path to JSON file

    Returns:
        Parsed dict or None if file doesn't exist or is invalid
    """
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load %s: %s", path, e)
    return None
