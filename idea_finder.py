#!/usr/bin/env python3
"""Old entry point. The code now lives in the idea_finder/ package.

Kept so that existing commands like `python idea_finder.py all ...` still work.
`python -m idea_finder ...` does the same thing.
"""

from idea_finder.cli import main

if __name__ == "__main__":
    main()
