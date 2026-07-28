# CuisineRAG 🍛

### A Modular Retrieval-Augmented Generation (RAG) System for South Asian Cuisine Knowledge

CuisineRAG is a modular **Retrieval-Augmented Generation (RAG)** system designed to answer questions about **South Asian cuisine** using a curated knowledge base. The system retrieves relevant contextual information from a dataset and uses a **Large Language Model (LLM)** to generate grounded responses.

---

## Quick Start

1. Open the main_notebook.ipynb , run all the cells one by one.
2. Place your input file (in json format) in the "inputs_and_outputs" folder.
3. You also have the option to change your output file location.
4. After executing the run_json_input_output cell, you can view your output from this location.
5. For evaluation on our benchmark dataset (which is named latest_benchmark.json and is placed in "data" folder) run the cell containing evaluate_rag_pipeline function.
6. The evaluation metrics along with the scores are printed in the output of this cell.
