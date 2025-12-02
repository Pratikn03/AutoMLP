#!/usr/bin/env python3
"""Backward-compatible wrapper for staging demo datasets.

Older docs referenced `scripts/download_datasets.py`; this module simply
re-exports the logic from `scripts/stage_datasets.py` so existing commands
continue to work.
"""

from stage_datasets import main


if __name__ == "__main__":
    main()
