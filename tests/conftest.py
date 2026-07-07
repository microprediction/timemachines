"""Dev convenience: if skaters isn't installed, use the sibling checkout."""
import os
import sys

try:
    import skaters  # noqa: F401
except ImportError:
    sib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "..", "skaters", "src")
    if os.path.isdir(sib):
        sys.path.insert(0, sib)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
