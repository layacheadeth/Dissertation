"""
Step 3: load the chunks into the database.

Run it:
    python 3-1-Ingest-to-ChromaDB-bge-Embed.py

Builds one collection per chunking strategy (exp1, exp2, exp3), all with
BGE, which is the only embedding model the project uses.

Each collection is rebuilt from scratch every run.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Ingestion_pipeline.ingestion_strategy import ingest_all

if __name__ == "__main__":
    ingest_all()
