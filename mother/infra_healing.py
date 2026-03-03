"""
Infrastructure self-healing — diagnose and repair broken infrastructure.

LEAF module (stdlib only). Detects and repairs missing DBs, corrupt configs,
stale locks, orphaned temp files. Mother recovers from common failure states
without user intervention.

Genome #80: infrastructure-healing — detects when services are degrading
before they fail.
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("mother.infra_healing")

# Default infrastructure paths
_DEFAULT_CONFIG_DIR = Path.home() / ".motherlabs"
_CONFIG_FILE = "mother.json"
_HISTORY_DB = "history.db"
_TOOLS_DB = "tools.db"
_LOCK_DIR = "locks"
_TEMP_DIR = "tmp"

# Thresholds
_STALE_LOCK_MINUTES = 30
_TEMP_FILE_MAX_AGE_HOURS = 24


@dataclass(frozen=True)
class HealthCheck:
    """Result of a single health check."""

    component: str          # "config", "history_db", "tools_db", "temp_files", "locks"
    status: str             # "healthy", "degraded", "broken"
    message: str
    auto_healable: bool


@dataclass
class HealingResult:
    """Result of a healing action."""

    component: str
    action: str             # "recreated", "repaired", "cleaned", "skipped"
    success: bool
    detail: str


class InfraHealer:
    """Diagnose and repair Mother's infrastructure."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or _DEFAULT_CONFIG_DIR

    def diagnose(self) -> List[HealthCheck]:
        """Run all health checks. Returns list of HealthCheck results."""
        checks = [
            self.check_config_dir(),
            self.check_config(),
            self.check_history_db(),
            self.check_tools_db(),
            self.check_temp_files(),
            self.check_stale_locks(),
        ]
        return checks

    def heal(self, checks: Optional[List[HealthCheck]] = None) -> List[HealingResult]:
        """Auto-heal what's fixable. Only acts on auto_healable=True.

        If checks is None, runs diagnose() first.
        """
        if checks is None:
            checks = self.diagnose()

        results: List[HealingResult] = []
        for check in checks:
            if check.status == "healthy":
                continue
            if not check.auto_healable:
                results.append(HealingResult(
                    component=check.component,
                    action="skipped",
                    success=False,
                    detail=f"Not auto-healable: {check.message}",
                ))
                continue

            result = self._heal_component(check)
            results.append(result)

        return results

    # --- Individual checks ---

    def check_config_dir(self) -> HealthCheck:
        """Check if config directory exists."""
        if self.config_dir.is_dir():
            return HealthCheck(
                component="config_dir",
                status="healthy",
                message=f"Config dir exists: {self.config_dir}",
                auto_healable=True,
            )
        return HealthCheck(
            component="config_dir",
            status="broken",
            message=f"Config dir missing: {self.config_dir}",
            auto_healable=True,
        )

    def check_config(self) -> HealthCheck:
        """Check if mother.json is valid."""
        config_path = self.config_dir / _CONFIG_FILE
        if not config_path.exists():
            return HealthCheck(
                component="config",
                status="degraded",
                message="Config file missing (will use defaults)",
                auto_healable=True,
            )

        try:
            text = config_path.read_text(encoding="utf-8").strip()
            if not text:
                return HealthCheck(
                    component="config",
                    status="broken",
                    message="Config file is empty",
                    auto_healable=True,
                )
            data = json.loads(text)
            if not isinstance(data, dict):
                return HealthCheck(
                    component="config",
                    status="broken",
                    message="Config file is not a JSON object",
                    auto_healable=True,
                )
            return HealthCheck(
                component="config",
                status="healthy",
                message=f"Config valid ({len(data)} keys)",
                auto_healable=True,
            )
        except json.JSONDecodeError as e:
            return HealthCheck(
                component="config",
                status="broken",
                message=f"Config JSON corrupt: {e}",
                auto_healable=True,
            )
        except OSError as e:
            return HealthCheck(
                component="config",
                status="broken",
                message=f"Config read error: {e}",
                auto_healable=True,
            )

    def check_history_db(self) -> HealthCheck:
        """Check if history.db is a valid SQLite database."""
        return self._check_db(_HISTORY_DB)

    def check_tools_db(self) -> HealthCheck:
        """Check if tools.db is a valid SQLite database."""
        return self._check_db(_TOOLS_DB)

    def check_temp_files(self) -> HealthCheck:
        """Check for orphaned temp files."""
        temp_dir = self.config_dir / _TEMP_DIR
        if not temp_dir.exists():
            return HealthCheck(
                component="temp_files",
                status="healthy",
                message="No temp directory",
                auto_healable=True,
            )

        now = time.time()
        max_age = _TEMP_FILE_MAX_AGE_HOURS * 3600
        stale_count = 0

        try:
            for f in temp_dir.iterdir():
                if f.is_file():
                    age = now - f.stat().st_mtime
                    if age > max_age:
                        stale_count += 1
        except OSError:
            pass

        if stale_count == 0:
            return HealthCheck(
                component="temp_files",
                status="healthy",
                message="No stale temp files",
                auto_healable=True,
            )
        return HealthCheck(
            component="temp_files",
            status="degraded",
            message=f"{stale_count} stale temp file(s) older than {_TEMP_FILE_MAX_AGE_HOURS}h",
            auto_healable=True,
        )

    def check_stale_locks(self) -> HealthCheck:
        """Check for stale lock files."""
        lock_dir = self.config_dir / _LOCK_DIR
        if not lock_dir.exists():
            return HealthCheck(
                component="locks",
                status="healthy",
                message="No lock directory",
                auto_healable=True,
            )

        now = time.time()
        max_age = _STALE_LOCK_MINUTES * 60
        stale_count = 0

        try:
            for f in lock_dir.iterdir():
                if f.is_file() and f.suffix == ".lock":
                    age = now - f.stat().st_mtime
                    if age > max_age:
                        stale_count += 1
        except OSError:
            pass

        if stale_count == 0:
            return HealthCheck(
                component="locks",
                status="healthy",
                message="No stale locks",
                auto_healable=True,
            )
        return HealthCheck(
            component="locks",
            status="degraded",
            message=f"{stale_count} stale lock(s) older than {_STALE_LOCK_MINUTES}min",
            auto_healable=True,
        )

    # --- Internal helpers ---

    def _check_db(self, db_name: str) -> HealthCheck:
        """Check if a SQLite database is valid."""
        db_path = self.config_dir / db_name
        component = db_name.replace(".db", "_db")

        if not db_path.exists():
            return HealthCheck(
                component=component,
                status="degraded",
                message=f"{db_name} missing (will be created on first use)",
                auto_healable=True,
            )

        try:
            conn = sqlite3.connect(str(db_path))
            # integrity_check reads actual pages — SELECT 1 is a constant
            # expression that succeeds even on corrupt files.
            result = conn.execute("PRAGMA integrity_check").fetchone()
            conn.close()
            if result[0] != "ok":
                raise sqlite3.DatabaseError(f"integrity_check: {result[0]}")
            return HealthCheck(
                component=component,
                status="healthy",
                message=f"{db_name} valid",
                auto_healable=True,
            )
        except sqlite3.DatabaseError as e:
            return HealthCheck(
                component=component,
                status="broken",
                message=f"{db_name} corrupt: {e}",
                auto_healable=True,
            )

    def _heal_component(self, check: HealthCheck) -> HealingResult:
        """Dispatch healing based on component name."""
        healers = {
            "config_dir": self._heal_config_dir,
            "config": self._heal_config,
            "history_db": lambda: self._heal_db(_HISTORY_DB),
            "tools_db": lambda: self._heal_db(_TOOLS_DB),
            "temp_files": self._heal_temp_files,
            "locks": self._heal_stale_locks,
        }

        healer = healers.get(check.component)
        if healer is None:
            return HealingResult(
                component=check.component,
                action="skipped",
                success=False,
                detail=f"No healer for component: {check.component}",
            )

        try:
            return healer()
        except Exception as e:
            logger.error(f"Healing failed for {check.component}: {e}")
            return HealingResult(
                component=check.component,
                action="skipped",
                success=False,
                detail=f"Healing error: {e}",
            )

    def _heal_config_dir(self) -> HealingResult:
        """Create missing config directory."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            return HealingResult(
                component="config_dir",
                action="recreated",
                success=True,
                detail=f"Created {self.config_dir}",
            )
        except OSError as e:
            return HealingResult(
                component="config_dir",
                action="skipped",
                success=False,
                detail=f"Failed to create config dir: {e}",
            )

    def _heal_config(self) -> HealingResult:
        """Recreate config with defaults."""
        config_path = self.config_dir / _CONFIG_FILE
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # If corrupt, back it up first
        if config_path.exists():
            backup = config_path.with_suffix(f".corrupt.{int(time.time())}")
            try:
                config_path.rename(backup)
                logger.info(f"Backed up corrupt config to {backup}")
            except OSError:
                pass

        try:
            default_config = {"schema_version": 1}
            config_path.write_text(json.dumps(default_config, indent=2), encoding="utf-8")
            return HealingResult(
                component="config",
                action="recreated",
                success=True,
                detail="Recreated config with defaults",
            )
        except OSError as e:
            return HealingResult(
                component="config",
                action="skipped",
                success=False,
                detail=f"Failed to recreate config: {e}",
            )

    def _heal_db(self, db_name: str) -> HealingResult:
        """Recreate a corrupt/missing database."""
        db_path = self.config_dir / db_name
        component = db_name.replace(".db", "_db")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Back up corrupt DB
        if db_path.exists():
            backup = db_path.with_suffix(f".corrupt.{int(time.time())}")
            try:
                db_path.rename(backup)
                logger.info(f"Backed up corrupt DB to {backup}")
            except OSError:
                pass

        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("SELECT 1")
            conn.close()
            return HealingResult(
                component=component,
                action="recreated",
                success=True,
                detail=f"Recreated {db_name}",
            )
        except sqlite3.Error as e:
            return HealingResult(
                component=component,
                action="skipped",
                success=False,
                detail=f"Failed to recreate {db_name}: {e}",
            )

    def _heal_temp_files(
        self,
        max_age_hours: int = _TEMP_FILE_MAX_AGE_HOURS,
    ) -> HealingResult:
        """Clean temp files older than threshold."""
        temp_dir = self.config_dir / _TEMP_DIR
        if not temp_dir.exists():
            return HealingResult(
                component="temp_files",
                action="skipped",
                success=True,
                detail="No temp directory to clean",
            )

        now = time.time()
        max_age = max_age_hours * 3600
        cleaned = 0

        try:
            for f in temp_dir.iterdir():
                if f.is_file():
                    age = now - f.stat().st_mtime
                    if age > max_age:
                        f.unlink()
                        cleaned += 1
        except OSError as e:
            logger.warning(f"Error cleaning temp files: {e}")

        return HealingResult(
            component="temp_files",
            action="cleaned",
            success=True,
            detail=f"Removed {cleaned} stale temp file(s)",
        )

    def _heal_stale_locks(
        self,
        max_age_minutes: int = _STALE_LOCK_MINUTES,
    ) -> HealingResult:
        """Break stale lock files."""
        lock_dir = self.config_dir / _LOCK_DIR
        if not lock_dir.exists():
            return HealingResult(
                component="locks",
                action="skipped",
                success=True,
                detail="No lock directory",
            )

        now = time.time()
        max_age = max_age_minutes * 60
        broken = 0

        try:
            for f in lock_dir.iterdir():
                if f.is_file() and f.suffix == ".lock":
                    age = now - f.stat().st_mtime
                    if age > max_age:
                        f.unlink()
                        broken += 1
                        logger.info(f"Broke stale lock: {f.name}")
        except OSError as e:
            logger.warning(f"Error breaking locks: {e}")

        return HealingResult(
            component="locks",
            action="cleaned",
            success=True,
            detail=f"Broke {broken} stale lock(s)",
        )
