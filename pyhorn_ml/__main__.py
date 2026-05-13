"""Wrapper script so 'pyhorn-ml' launches correctly.

The shebang Python may not have the editable install on sys.path
(the site.addsitedir for .pth files runs once and the path hook
mechanism can shadow co-installed packages). This wrapper fixes sys.path
BEFORE any pyhorn_ml imports so the package is always findable.
"""
import sys
import os

# Ensure project root is on sys.path so pyhorn-ml is always importable
# regardless of how Python was invoked.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pyhorn_ml.cli.commands import app

if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    app()
