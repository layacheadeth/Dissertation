import os
import json
import glob
import chromadb

# ------------------------------------------------------------------
# 1. Path & Strategy Configurations
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data") if os.path.exists(os.path.join(SCRIPT_DIR, "Data")) else SCRIPT_DIR
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma_db")

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Strategy definitions mapping file patterns to collection names
STRATEGIES = {
    "exp1": {
        "file_pattern": "exp1_chunks.json",
        "collection_name": "exp1_page_level",
        "description": "Page-Level Chunking Strategy"
    },
    "exp2": {
        "file_pattern": "exp2_chunks.json",
        "collection_name": "exp2_fixed_overlap",
        "description": "Fixed-Size Overlapping Chunking Strategy"
    },
    "exp3": {
        "file_pattern": "exp3_chunks.json",
        "collection_name": "exp3_structure_level",
        "description": "Structure-Level Chunking Strategy"
    }
}

# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------
def extract_text_and_meta(item, default_source, default_week, strategy_key):
    """
    Extracts text content and normalizes metadata for ChromaDB.
    """
    text = (
        item.get("text") or 
        item.get("chunk") or 
        item.get("content") or 
        item.get("page_content") or 
        ""
    )
    if isinstance(text, dict):
        text = json.dumps(text)
        
    metadata = {
        "week": str(item.get("week", default_week)),
        "source_file": str(item.get("source_file", default_source)),
        "page": str(item.get("page", item.get("page_number", "N/A"))),
        "chunking_strategy": strategy_key
    }
    
    # Preserve additional scalar metadata
    for k, v in item.items():
        if k not in ["text", "chunk", "content", "page_content"] and isinstance(v, (str, int, float, bool)):
            metadata[k] = str(v)
            
    return text.strip(), metadata


# ------------------------------------------------------------------
# 3. Strategy-by-Strategy Ingestion
# ------------------------------------------------------------------
def ingest_strategy(strat_key, config):
    collection_name = config["collection_name"]
    pattern = config["file_pattern"]
    
    print(f"\n==================================================")
    print(f"🚀 Ingesting Strategy [{strat_key.upper()}] -> Collection: '{collection_name}'")
    print(f"==================================================")
    
    # Create or get dedicated collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": config["description"]}
    )
    
    documents = []
    metadatas = []
    ids = []
    seen_ids = set()
    
    # Find all weekly directories
    week_folders = sorted(glob.glob(os.path.join(DATA_DIR, "Data_week*")))
    
    for folder_path in week_folders:
        week_name = os.path.basename(folder_path)
        
        # Match only files for this specific strategy across all weeks
        target_files = glob.glob(os.path.join(folder_path, f"*{pattern}*"))
        
        for json_path in target_files:
            file_name = os.path.basename(json_path)
            
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                items = data if isinstance(data, list) else [data]
                
                for idx, item in enumerate(items):
                    text, meta = extract_text_and_meta(item, file_name, week_name, strat_key)
                    
                    if not text:
                        continue
                    
                    # Generate unique ID per document within this batch
                    raw_id = item.get("chunk_id") or item.get("id") or f"{week_name}_{idx}"
                    doc_id = f"{strat_key}_{raw_id}"
                    
                    # Deduplicate in-memory to prevent DuplicateIDError
                    dup_cnt = 1
                    base_id = doc_id
                    while doc_id in seen_ids:
                        doc_id = f"{base_id}_dup{dup_cnt}"
                        dup_cnt += 1
                        
                    seen_ids.add(doc_id)
                    documents.append(text)
                    metadatas.append(meta)
                    ids.append(doc_id)
                    
            except Exception as e:
                print(f"❌ Error loading {json_path}: {e}")

    # Upsert in batches of 500
    if documents:
        print(f"Found {len(documents)} total chunks for {strat_key.upper()}. Upserting into '{collection_name}'...")
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size]
            )
        print(f" Successful ingestion into '{collection_name}'!")
    else:
        print(f"⚠️ No chunks found matching pattern: {pattern}")


# ------------------------------------------------------------------
# 4. Main Execution Routine
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Loop and process each strategy independently
    for strat_key, config in STRATEGIES.items():
        ingest_strategy(strat_key, config)
        
    print("\n🎉 All 3 strategy collections created successfully in ChromaDB!")