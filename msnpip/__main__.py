"""Enable ``python -m msnpip ...`` as an alias for the ``msnpip`` console script."""

from __future__ import annotations

import sys

from msnpip.cli import main

if __name__ == "__main__":
    sys.exit(main())
