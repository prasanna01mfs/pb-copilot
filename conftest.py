"""Put the project root on sys.path so tests can `import tools`, `harness`, etc.

Lets both `pytest` and `python -m pytest` work from the project root without an
installed package or PYTHONPATH juggling.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
