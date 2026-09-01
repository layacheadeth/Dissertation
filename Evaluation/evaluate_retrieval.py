"""
A browser front end for the local EduBot pipeline.
Optimized for container/workspace environments and older Gradio versions.
"""

import argparse
import socket
import sys
from pathlib import Path
import gradio as gr

# Ensure sibling modules can be imported
HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE / "Inference_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Inference_pipeline.RAG_pipeline import add_pipeline_flags, build_pipeline


def find_free_port(starting_port=7860):
    port = starting_port
    while port < starting_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('0.0.0.0', port)) != 0:
                return port
            port += 1
    return starting_port


def format_chunks(documents, scores=None):
    if not documents:
        return "_No slides were retrieved._"

    scores = scores or [None] * len(documents)
    lines = []
    
    for i, (doc, score) in enumerate(zip(documents, scores), start=1):
        meta = doc.metadata or {}
        week = meta.get("week") or meta.get("week_number") or "?"
        page = meta.get("page_number", "?")
        
        header = f"**{i}. {week}, page {page}**"
        if score is not None:
            header += f"  _(relevance {score:.2f})_"
            
        body = doc.page_content.strip()
        if len(body) > 600:
            body = body[:600].rstrip() + " …"
            
        lines.append(f"{header}\n\n> {body}")
        
    return "\n\n---\n\n".join(lines)


def build_ui(pipeline, settings_line):
    def answer(question, history):
        question = (question or "").strip()
        if not question:
            return history, "_Ask a question about the lecture material._"

        plan = pipeline.run(question)

        note = {
            "meta": "_Answered without searching: question about the system._",
            "refused": "_Refused: nothing retrieved cleared the relevance floor._"
        }
        
        history = history or []
        history.append((question, str(plan.answer)))
        
        sources = note.get(plan.route) or format_chunks(plan.documents, plan.scores)
        return history, sources

    with gr.Blocks(title="EduBot - Local Course Assistant") as demo:
        gr.Markdown(f"# EduBot: COMP64702 Assistant\n**Active Configuration:** `{settings_line}`")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Removed height and type args entirely for maximum compatibility
                chat = gr.Chatbot()
                box = gr.Textbox(placeholder="e.g. What is BM25?", label="Question")
                with gr.Row():
                    send = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")
            
            with gr.Column(scale=1):
                sources = gr.Markdown(label="Slides used", value="_Waiting for a question..._")

        send.click(answer, [box, chat], [chat, sources]).then(lambda: "", None, box)
        box.submit(answer, [box, chat], [chat, sources]).then(lambda: "", None, box)
        clear.click(lambda: ([], "_Waiting for a question..._"), None, [chat, sources])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Local EduBot UI")
    add_pipeline_flags(parser)
    parser.add_argument("--share", action="store_true", help="expose a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    target_port = find_free_port(args.port)

    print("\n[Step 1/2] Loading AI Models into memory...")
    pipeline, search, retriever = build_pipeline(
        strategy=args.strategy,
        combo=args.combo,
        embedder=args.embedder,
        llm_name=args.llm,
        candidates=args.candidates,
        top_n=args.top_n,
        budget_tokens=args.budget_tokens,
        device=args.device,
        verify=not args.no_verify,
        route_meta=not args.no_router,
        relevance_floor=args.relevance_floor,
    )

    cut = f"budget_tokens={args.budget_tokens}" if args.budget_tokens else f"top_n={args.top_n}"
    settings = f"{args.strategy} | {args.embedder} | {args.combo} | {args.llm} | {cut}"
    
    print("\n[Step 2/2] Launching Gradio Web Server...")
    demo = build_ui(pipeline, settings)
    
    print(f"\n==================================================")
    print(f" EduBot is live! Check your workspace's port forwarding tab.")
    print(f" (Running on 0.0.0.0:{target_port})")
    print(f"==================================================\n")
    
    # Changed to 0.0.0.0 so remote workspaces can map the port properly
    demo.launch(server_name="0.0.0.0", server_port=target_port, share=args.share)


if __name__ == "__main__":
    main()