"""
Legacy preservation — schema versioning, migration, and backward compatibility.

LEAF module (stdlib only). Ensures Mother can read/migrate data from
older versions without silent breakage.

Genome #18: legacy-preserving — when an instance dies, its learnings
persist in the parent. Config format changes don't corrupt data.
"""

import copy
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("mother.legacy")

CURRENT_SCHEMA_VERSION = 1


@dataclass
class MigrationResult:
    """Result of a schema migration."""

    from_version: int
    to_version: int
    changes: List[str]
    success: bool
    backup_path: Optional[Path] = None


class LegacyError(Exception):
    """Legacy migration error."""


def _migrate_v0_to_v1(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Migrate from v0 (no version) to v1.

    Changes:
    - Add schema_version field
    - Normalize api_keys to dict if missing
    - Ensure output_dir is string
    """
    changes: List[str] = []
    result = copy.deepcopy(data)

    # Add schema version
    result["schema_version"] = 1
    changes.append("Added schema_version=1")

    # Normalize api_keys
    if "api_keys" not in result:
        result["api_keys"] = {}
        changes.append("Added empty api_keys dict")
    elif not isinstance(result.get("api_keys"), dict):
        result["api_keys"] = {}
        changes.append("Reset invalid api_keys to empty dict")

    # Ensure output_dir is string
    if "output_dir" in result and not isinstance(result["output_dir"], str):
        result["output_dir"] = str(result["output_dir"])
        changes.append("Converted output_dir to string")

    return result, changes


# Migration registry: version -> (migration_function)
# Each function takes dict, returns (dict, list[str])
_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Tuple[Dict[str, Any], List[str]]]] = {
    0: _migrate_v0_to_v1,
}


class LegacyGuard:
    """Version-aware config/data reader with migration support."""

    @staticmethod
    def read_config(path: Path) -> Tuple[Dict[str, Any], int]:
        """Read config file, return (data, schema_version).

        Missing version field treated as version 0.
        Missing/empty file returns empty dict with version 0.
        """
        if not path.exists():
            return {}, 0

        try:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                return {}, 0
            data = json.loads(text)
            if not isinstance(data, dict):
                logger.warning(f"Config is not a dict: {path}")
                return {}, 0
            version = data.get("schema_version", 0)
            if not isinstance(version, int):
                version = 0
            return data, version
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt config JSON at {path}: {e}")
            return {}, 0

    @staticmethod
    def needs_migration(data: Dict[str, Any]) -> bool:
        """True if data needs migration to current schema version."""
        version = data.get("schema_version", 0)
        if not isinstance(version, int):
            return True
        return version < CURRENT_SCHEMA_VERSION

    @staticmethod
    def migrate(data: Dict[str, Any], from_version: int) -> MigrationResult:
        """Apply migrations sequentially from from_version to CURRENT_SCHEMA_VERSION.

        Raises LegacyError if from_version > CURRENT_SCHEMA_VERSION (future version).
        """
        if from_version > CURRENT_SCHEMA_VERSION:
            raise LegacyError(
                f"Config version {from_version} is newer than supported "
                f"version {CURRENT_SCHEMA_VERSION}. Cannot downgrade."
            )

        if from_version == CURRENT_SCHEMA_VERSION:
            return MigrationResult(
                from_version=from_version,
                to_version=CURRENT_SCHEMA_VERSION,
                changes=["Already at current version"],
                success=True,
            )

        current_data = copy.deepcopy(data)
        all_changes: List[str] = []
        current_version = from_version

        while current_version < CURRENT_SCHEMA_VERSION:
            migration_fn = _MIGRATIONS.get(current_version)
            if migration_fn is None:
                return MigrationResult(
                    from_version=from_version,
                    to_version=current_version,
                    changes=all_changes + [
                        f"No migration found for v{current_version}→v{current_version + 1}"
                    ],
                    success=False,
                )

            try:
                current_data, changes = migration_fn(current_data)
                all_changes.extend(changes)
                current_version += 1
            except Exception as e:
                logger.error(f"Migration v{current_version}→v{current_version + 1} failed: {e}")
                return MigrationResult(
                    from_version=from_version,
                    to_version=current_version,
                    changes=all_changes + [f"Migration failed: {e}"],
                    success=False,
                )

        return MigrationResult(
            from_version=from_version,
            to_version=CURRENT_SCHEMA_VERSION,
            changes=all_changes,
            success=True,
        )

    @staticmethod
    def backup_before_migration(path: Path) -> Path:
        """Create a timestamped backup of a file before migration.

        Returns the backup path.
        Raises FileNotFoundError if source doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Cannot backup: {path} does not exist")

        timestamp = int(time.time())
        backup_path = path.with_suffix(f".bak.{timestamp}")
        shutil.copy2(str(path), str(backup_path))
        logger.info(f"Backup created: {backup_path}")
        return backup_path

    @staticmethod
    def ensure_schema_version(data: Dict[str, Any]) -> Dict[str, Any]:
        """Stamp current schema version if missing. Non-destructive.

        Returns a new dict (does not mutate input).
        """
        result = copy.deepcopy(data)
        if "schema_version" not in result:
            result["schema_version"] = CURRENT_SCHEMA_VERSION
        return result

    @staticmethod
    def migrate_file(path: Path, dry_run: bool = False) -> MigrationResult:
        """Read, backup, migrate, and write config file.

        If dry_run=True, does not write or backup.
        """
        data, version = LegacyGuard.read_config(path)

        if not LegacyGuard.needs_migration(data):
            return MigrationResult(
                from_version=version,
                to_version=version,
                changes=["No migration needed"],
                success=True,
            )

        result = LegacyGuard.migrate(data, version)

        if result.success and not dry_run:
            backup_path = LegacyGuard.backup_before_migration(path)
            result.backup_path = backup_path

            # Write migrated data
            migrated = copy.deepcopy(data)
            # Re-run migration on original to get final state
            migrated_result = LegacyGuard.migrate(data, version)
            # Apply changes by re-running migration
            current = copy.deepcopy(data)
            current_v = version
            while current_v < CURRENT_SCHEMA_VERSION:
                fn = _MIGRATIONS.get(current_v)
                if fn:
                    current, _ = fn(current)
                    current_v += 1
                else:
                    break

            path.write_text(json.dumps(current, indent=2), encoding="utf-8")
            logger.info(f"Migrated {path}: v{version} → v{CURRENT_SCHEMA_VERSION}")

        return result
