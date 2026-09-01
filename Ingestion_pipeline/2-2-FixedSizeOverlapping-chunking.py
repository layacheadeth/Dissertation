"""
Step 2b: fixed-size overlapping chunking.

Run it:
    python 2-2-FixedSizeOverlapping-chunking.py

A fixed-size window slides across the whole lecture, ignoring where one slide
ends and the next begins. Neighbouring chunks share some text.

Reads  Data/All_extracted_text/Data_week*/<lecture>.json
Writes Data/All_extracted_text/Data_week*/<lecture>_exp2_chunks.json
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Share_components.chunking_strategies import chunk_all

if __name__ == "__main__":
    chunk_all("exp2")
