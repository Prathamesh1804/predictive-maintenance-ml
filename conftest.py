"""
Pytest configuration — adds the project root to sys.path.
"""
import sys
from pathlib import Path

# Ensure project root is always on sys.path for all tests
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
