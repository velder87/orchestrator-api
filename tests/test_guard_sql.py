import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import guard_sql


def test_guard_sql_allows_simple_select():
    # Should not raise for safe read-only query
    guard_sql("SELECT * FROM products")


def test_guard_sql_rejects_semi_colon_injection():
    with pytest.raises(HTTPException) as exc:
        guard_sql("SELECT 1; DROP TABLE users")

    assert exc.value.status_code == 400
    assert "read-only" in exc.value.detail


def test_guard_sql_rejects_mutating_keywords():
    with pytest.raises(HTTPException) as exc:
        guard_sql("DeLeTe FROM audit_log WHERE id = 1")

    assert exc.value.status_code == 400
    assert "read-only" in exc.value.detail
