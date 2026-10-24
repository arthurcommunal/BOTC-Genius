"""Writes a tenants file for the tests.

Test keys are fixed (not random) so the tests know them in clear text;
only their hashes are written to the file, like in production.

Usage as a script: uv run python tests/data/generate_test_json.py <output_path>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from botc.serving.api.auth import KEY_PREFIX, hash_api_key

USER_KEY = f"{KEY_PREFIX}test_user"
ADMIN_KEY = f"{KEY_PREFIX}test_admin"

TENANTS = [
    {"tenant_id": "user", "key": USER_KEY, "rate": 1.0, "capacity": 5, "is_admin": False},
    {"tenant_id": "admin", "key": ADMIN_KEY, "rate": 1.0, "capacity": 5, "is_admin": True},
]


def write_tenants_json(path: Path) -> Path:
    """Writes the tenants of `TENANTS` to `path` (hashes only) and returns `path`."""
    rows = []
    for tenant in TENANTS:
        row = {k: v for k, v in tenant.items() if k != "key"}
        row["key_hash"] = hash_api_key(tenant["key"])
        rows.append(row)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/data/generate_test_json.py <output_path>")
    print(write_tenants_json(Path(sys.argv[1])))
