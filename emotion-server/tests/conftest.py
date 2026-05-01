"""
Pytest config: import the bits of server.py we want to test without triggering
the heavy imports (torch, whisperx, mediapipe). We do this by stubbing those
modules in sys.modules BEFORE server.py is imported.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub(name: str) -> types.ModuleType:
    """Insert a fake module so `import name` succeeds without the real package."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# Stub heavy / GPU-only imports.
if "torch" not in sys.modules:
    torch_mod = _stub("torch")
    torch_mod.cuda = MagicMock()
    torch_mod.cuda.is_available = lambda: False

if "whisperx" not in sys.modules:
    whisperx_mod = _stub("whisperx")
    whisperx_mod.load_model = MagicMock(return_value=MagicMock())
    whisperx_mod.load_audio = MagicMock()
