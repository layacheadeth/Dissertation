"""
Step 2d: section-aware chunking.

Run it:
    python 2-3-SectionAware-chunking.py

Slides that share a heading are merged into one section, and the heading is
put at the top of every chunk from that section.

Reads  Data/All_extracted_text/Data_week*/<lecture>.json
Writes Data/All_extracted_text/Data_week*/<lecture>_exp3_chunks.json
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Share_components.chunking_strategies import chunk_all

if __name__ == "__main__":
    chunk_all("exp3")
