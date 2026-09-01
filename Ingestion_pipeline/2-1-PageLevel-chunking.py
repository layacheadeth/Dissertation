"""
Step 2a: page-level chunking.

Run it:
    python 2-1-PageLevel-chunking.py

One chunk per slide. A slide too long for the model is cut into parts.

Reads  Data/All_extracted_text/Data_week*/<lecture>.json
Writes Data/All_extracted_text/Data_week*/<lecture>_exp1_chunks.json
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Share_components.chunking_strategies import chunk_all

if __name__ == "__main__":
    chunk_all("exp1")
