"""Configuração global de testes: garante que a raiz do repo está no PYTHONPATH."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
