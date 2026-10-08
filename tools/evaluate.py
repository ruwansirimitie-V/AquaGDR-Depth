"""Convenience entry point for AquaGDR-Depth evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquagdr.eval_fusion import main


if __name__ == "__main__":
    main()
