import ast
from pathlib import Path


def test_dashboard_app_parses():
    src = Path("dashboard/app.py").read_text(encoding="utf-8")
    ast.parse(src)
