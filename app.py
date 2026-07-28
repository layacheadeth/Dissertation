import gradio as gr
import torch
import json
import os
from Inference_pipeline.chunking import SectionAwareChunker
from Inference_pipeline.embeddings import MiniLMEmbedding
from vectore_store import FAISSVectorDB
from Inference_pipeline.ranking_n_retrieval import Retriever
from Inference_pipeline.llm_n_prompt import QwenLLM, PromptTemplate, classify_intent
from Inference_pipeline.rag_pipeline import RAGPipeline

# --- PIPELINE INITIALIZATION ---
def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    else: return "cpu"

print("Initializing RAG Pipeline Components for Gradio UI...")
DEVICE = get_device()
INDEX_BIN = "data/index.bin"
DOCS_JSON = "data/docs.json"
FILEPATHS = ["Data_week1/comp64702_week1.json",
             "Data_week2/COMP64702_week2.json",
             "Data_week3/comp34711_week3.json",
             "Data_week4/corpus/comp61111_week4.json",
             "Data_week5/trim_week5.json",
             "Data_week6/trim_week6.json",
             "Data_week7/trim_week7.json",
             "Data_week8/corpus/comp61332_week8.json",
             "Data_week9/comp9412_week9.json",
             "Data_week10/TM_week10.json",
             "Data_week11/comp34812_week11.json",
             "Data_week12/tm_week12.json",
             "Data_week13/comp64702_week13.json"
             ]

# Build singletons matching your production configuration
chunker = SectionAwareChunker()
embedder, _ = MiniLMEmbedding(), 384
vectordb = FAISSVectorDB(dim=384)
prompt_builder = PromptTemplate()
llm = QwenLLM(device=DEVICE)

# Setup pipeline container
pipeline = RAGPipeline(chunker, embedder, vectordb, None, prompt_builder, llm)

# Force load or build index
if os.path.exists(INDEX_BIN) and os.path.exists(DOCS_JSON):
    vectordb.load(INDEX_BIN, DOCS_JSON)
    pipeline.chunks = vectordb.documents
    print("Loaded existing index state.")
else:
    print("No index cache found. Re-indexing raw COMP64702 corpus...")
    pipeline.index_data(FILEPATHS)
    os.makedirs("data", exist_ok=True)
    vectordb.save(INDEX_BIN, DOCS_JSON)

# Attach Retriever
retriever = Retriever(vectordb, pipeline.chunks, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever.active_combo = "combo2"  # Hybrid + Reranking
pipeline.retriever = retriever

# --- MODE OPTIONS ---
# Labels shown in the UI, mapped to how the engine should behave.
MODE_DYNAMIC = "Dynamic (Auto-Classify)"
MODE_QA = "QA Only"
MODE_SOCRATIC = "Socratic Only"

# --- GRADIO INTERACTION ENGINE ---
def socratic_chat_engine(user_query, history, mode_selection):
    if not user_query.strip():
        return "", history, ""

    # 1. RESOLVE THE MODE BASED ON THE UI SELECTION
    #    - Dynamic  -> run the classifier to pick "qa" or "socratic" per query
    #    - QA Only  -> force "qa" and skip classification
    #    - Socratic -> force "socratic" and skip classification
    if mode_selection == MODE_QA:
        mode = "qa"
        print("--- [ROUTER] Forced mode: QA ---")
    elif mode_selection == MODE_SOCRATIC:
        mode = "socratic"
        print("--- [ROUTER] Forced mode: SOCRATIC ---")
    else:  # MODE_DYNAMIC (default / fallback)
        mode = classify_intent(
            user_query,
            pipeline.llm.tokenizer,
            pipeline.llm.model,
            DEVICE
        )
        print(f"--- [ROUTER] Dynamic classification -> {mode.upper()} ---")

    # 2. PASS THE RESOLVED MODE INTO YOUR PIPELINE
    # This tells the prompt generator which system persona to use
    response, retrieved_docs = pipeline.query(user_query, mode=mode)

    # 3. FORMAT THE CONTEXT PREVIEW FOR THE INTERFACE
    context_details = ""
    for idx, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        context_details += f"### Chunk {idx+1} | Source: {meta.get('doc_id', '?')} ({meta.get('section_title', 'General')})\n"
        context_details += f"*{doc.page_content.strip()}*\n\n---\n\n"

    # Append user question and assistant response as explicit dicts for Gradio 4+
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response})

    return "", history, context_details

# --- INTERFACE DESIGN (BLOCKS) ---
with gr.Blocks(title="EduBot: COMP64702 Socratic Tutor") as demo:
    gr.Markdown("# 🎓 EduBot: COMP64702 Socratic AI Tutor")
    gr.Markdown("Interact with your hybrid RAG system running live inside Docker. EduBot uses the Socratic method to guide your learning using text representation contexts.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Socratic Conversation History",
                height=500
            )  # Gradio 6 uses the messages (role/content) format by default

            # NEW: tutor-mode selector
            mode_selector = gr.Radio(
                choices=[MODE_DYNAMIC, MODE_QA, MODE_SOCRATIC],
                value=MODE_DYNAMIC,
                label="🧭 Tutor Mode",
                info="Dynamic auto-detects intent per question. QA Only and Socratic Only lock the tutor into a single persona and skip the classifier."
            )

            msg = gr.Textbox(
                label="Ask a question about the course, vectors, or formatting...",
                placeholder="e.g., How is COMP64702 assessed?",
                show_label=True
            )
            clear = gr.Button("Clear History")

        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Retrieved Context Layer")
            context_box = gr.Markdown(
                value="*Submit a question to see the underlying text chunks retrieved by the Hybrid RRF + Cross-Encoder Reranker layer.*",
            )

    # Bind active state events (mode_selector is now passed into the engine)
    msg.submit(socratic_chat_engine, [msg, chatbot, mode_selector], [msg, chatbot, context_box])
    clear.click(lambda: (None, None, "*Context cleared.*"), None, [chatbot, context_box])

if __name__ == "__main__":
    # Server configuration maps to listen across Docker boundaries
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)