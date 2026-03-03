"""Huey task queue configuration — SqliteHuey backend.

Uses MOTHERLABS_DATA_DIR for database location (default: /data in Docker).
Set MOTHERLABS_HUEY_IMMEDIATE=1 for synchronous execution in tests.
"""

import os
from huey import SqliteHuey

_data_dir = os.environ.get("MOTHERLABS_DATA_DIR", "/data")
_db_path = os.path.join(_data_dir, "huey.db")
_immediate = os.environ.get("MOTHERLABS_HUEY_IMMEDIATE", "").strip() == "1"

huey = SqliteHuey(
    name="motherlabs",
    filename=_db_path,
    immediate=_immediate,
)
