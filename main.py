import torch
import json
from Inference_pipeline.chunking import SectionAwareChunker, SemanticChunker
from Inference_pipeline.embeddings import MiniLMEmbedding, QwenEmbedding
from vectore_store import FAISSVectorDB
from Inference_pipeline.ranking_n_retrieval import Retriever
from Inference_pipeline.llm_n_prompt import QwenLLM, PromptTemplate
from Inference_pipeline.rag_pipeline import RAGPipeline


# ==============================================================
# CONFIGURATION — change these to test different combinations
# ==============================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

CHUNKER    = "section"  # "section" or "semantic"
EMBEDDING  = "minilm"   # "minilm"  or  "qwen"
VECTORDB   = "faiss"    # "faiss"   or  "chroma"
RETRIEVAL  = "combo2"   # "combo1"  or  "combo2"
DEVICE     = get_device()
# Index the CURATED chunk indices (not the raw corpus). These carry the same
# chunk_ids the benchmark references, so runtime chunk_ids == benchmark
# relevant_chunk_ids, enabling true chunk-level retrieval evaluation.
FILEPATHS = ["Data_week1/chunks_index.json",
             "Data_week2/chunks_index.json",
             "Data_week3/chunks_index.json",
             "Data_week4/chunks_index.json",
             "Data_week5/chunks_index.json",
             "Data_week6/chunks_index.json",
             "Data_week7/chunks_index.json",
             "Data_week8/chunks_index.json",
             "Data_week9/chunks_index.json",
             "Data_week10/chunks_index.json",
             "Data_week11/chunks_index.json",
             "Data_week12/chunks_index.json",
             "Data_week13/chunks_index.json"
             ]
# ==============================================================


def build_embedder(choice):
    if choice == "minilm":
        return MiniLMEmbedding(), 384
    elif choice == "qwen":
        return QwenEmbedding(), 1024
    else:
        raise ValueError(f"Unknown embedder: {choice}. Choose 'minilm' or 'qwen'")


def build_vectordb(choice, dim):
    if choice == "faiss":
        return FAISSVectorDB(dim=dim)
    # elif choice == "chroma":
    #     return ChromaVectorDB()
    else:
        raise ValueError(f"Unknown vectordb: {choice}. Choose 'faiss' or 'chroma'")


def build_retriever(choice, vectordb, documents):
    retriever = Retriever(vectordb, documents)
    retriever.active_combo = choice   # store choice on the object
    return retriever


def run_json_input_output():

    print("\n" + "="*50)
    print("       Socratic_AI-TeachingAssistant")
    print("="*50)
    print(f"  Chunker: {CHUNKER}")
    print(f"  Embedding : {EMBEDDING}")
    print(f"  VectorDB  : {VECTORDB}")
    print(f"  Retrieval : {RETRIEVAL}")
    print(f"  Device    : {DEVICE}")
    print("="*50 + "\n")

    # --- build components based on config ---
    if CHUNKER == "semantic":
        chunker = SemanticChunker()       # reuses all-MiniLM-L6-v2
    else:
        chunker = SectionAwareChunker()
    embedder, dim = build_embedder(EMBEDDING)
    vectordb      = build_vectordb(VECTORDB, dim)
    prompt_builder = PromptTemplate()
    llm           = QwenLLM(device=DEVICE)

    # --- build pipeline (retriever added after indexing) ---

    pipeline = RAGPipeline(
        chunker        = chunker,
        embedder       = embedder,
        vectordb       = vectordb,
        retriever      = None,
        prompt_builder = prompt_builder,
        llm            = llm
    )

    # --- index the curated chunk indices ---

    import os

    # Fresh cache names so the old corpus-rechunked index is never reused.
    INDEX_BIN  = "faiss_index_chunks.bin"
    DOCS_JSON  = "faiss_docs_chunks.json"

    if os.path.exists(INDEX_BIN) and os.path.exists(DOCS_JSON):
        # already indexed — just load from disk
        vectordb.load(INDEX_BIN, DOCS_JSON)
        pipeline.chunks = vectordb.documents
        print("Loaded existing chunk index from disk.")
    else:
        # first run — index the curated chunks and save
        pipeline.index_chunks(FILEPATHS)
        vectordb.save(INDEX_BIN, DOCS_JSON)

    # --- build retriever with chunks for BM25 ---

    retriever = build_retriever(RETRIEVAL, vectordb, pipeline.chunks)
    pipeline.retriever = retriever

    input_file_path = "inputs_and_outputs/input_qa.json"
    output_file_path = "inputs_and_outputs/output_qa.json"
    with open(input_file_path) as input_file:
        query_list=json.load(input_file)['queries']

    print(f"\nProcessing {len(query_list)} queries from {input_file_path}...\n")

    results=[0]*len(query_list)
    for i, query in enumerate(query_list):
        query_id   = query["query_id"]
        query_text = query["query"].strip()
        mode       = query.get("intent", "qa")   # use per-query `intent` (qa/socratic); default qa
        # mode = "qa" # force QA for now to evaluate QA first. No work on socratic just yet. The above code does that though, but later.
        print(f"[{i+1}/{len(query_list)}] {query_text[:60]}...")
        answer, docs = pipeline.query(query_text, mode=mode)   # forward it
        results[i]={
            "query_id":str(query_id),
            "query":str(query_text),
            "response":str(answer),
            "retrieved_context":[{"doc_id":str(chunk.metadata.get('chunk_id', '?')),"text":chunk.page_content} for chunk in docs]
            }
        final=dict({"results": results})
    with open(output_file_path,'w') as output_json:
        json.dump(final, output_json, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(results)} results saved to {output_file_path}")



if __name__ == "__main__":
    run_json_input_output()