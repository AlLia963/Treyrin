"""R1 standard Treyrin model package (order 000064).

Import order: the package puts the inherited workspace and this directory on
sys.path, then the configuration and model classes are imported under their
standard HF module names (configuration_treyrin / modeling_treyrin) so that
class identity is the same whether the package is imported or the modules are
imported directly.
"""
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
_WORKSPACE = _MODEL_DIR.parent
for _path in (str(_WORKSPACE), str(_MODEL_DIR)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_WORKSPACE))
sys.path.insert(0, str(_MODEL_DIR))

from treyrin_paths import MODEL_DIR, WORKSPACE, bootstrap  # noqa: E402

bootstrap()

from configuration_treyrin import TreyrinConfig  # noqa: E402
from modeling_treyrin import TreyrinForCausalLM  # noqa: E402
from treyrin_assembly import (  # noqa: E402
    TreyrinAssembly, TreyrinStrictLoadError, assemble_workspace, key_report,
    load_standard_package, open_model, save_standard_package,
)

__all__ = [
    "MODEL_DIR", "WORKSPACE", "TreyrinConfig", "TreyrinForCausalLM",
    "TreyrinAssembly", "TreyrinStrictLoadError", "assemble_workspace",
    "load_standard_package", "save_standard_package", "open_model", "key_report",
]
