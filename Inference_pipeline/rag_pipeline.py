import json
from langchain_core.documents import Document

class RAGPipeline:

    def __init__(self, chunker, embedder, vectordb, retriever, prompt_builder,
                 llm_qa, llm_socratic=None):
        """Two LLMs: one for 'qa', one for 'socratic'. Pass a single llm to
        llm_qa (leaving llm_socratic=None) to reuse it for both modes."""
        self.chunker = chunker
        self.embedder = embedder
        self.vectordb = vectordb
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm_qa = llm_qa
        self.llm_socratic = llm_socratic if llm_socratic is not None else llm_qa


    def index_chunks(self, chunk_index_paths):
        """Index pre-built curated chunks from one or more chunks_index.json files.

        Unlike index_data (which re-chunks the raw corpus), this loads the SAME
        chunk boundaries and chunk_ids used to build the benchmark, so runtime
        chunk_ids match the benchmark's relevant_chunk_ids exactly — enabling
        true chunk-level retrieval evaluation.

        The stored embeddings inside chunks_index.json are ignored; the chunk
        text is re-embedded with the runtime embedder so the vector space is
        consistent with query embeddings.

        Args:
            chunk_index_paths: a single path string or a list of path strings.
        """
        if isinstance(chunk_index_paths, str):
            chunk_index_paths = [chunk_index_paths]

        # 1. Load and normalise every chunks_index.json (handles both shapes:
        #    a flat list, or a dict wrapping the list under a "chunks" key).
        records = []
        for fp in chunk_index_paths:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunk_list = data['chunks'] if isinstance(data, dict) else data
            records.extend(chunk_list)
            print(f"Loaded {len(chunk_list):,} curated chunks from {fp}")

        if not records:
            raise ValueError("All chunk indices are empty!")

        # 2. Build Document objects, preserving the curated chunk_id / doc_id.
        self.chunks = []
        for rec in records:
            text = rec.get("text", "")
            if not text or not isinstance(text, str):
                continue
            meta = {
                "chunk_id":      str(rec.get("chunk_id", "?")),
                "doc_id":        str(rec.get("doc_id", "?")),
                "section_title": rec.get("section_title", ""),
                "part":          rec.get("part", ""),
                "module_code":   rec.get("module_code", ""),
                "lecture_title": rec.get("lecture_title", ""),
            }
            self.chunks.append(Document(page_content=text.strip(), metadata=meta))

        # 3. Re-embed chunk text with the runtime embedder (not the stored vectors).
        texts = [doc.page_content for doc in self.chunks]
        embeddings = self.embedder.embed_documents(texts)

        # 4. Store into the vector DB.
        self.vectordb.add_documents(embeddings, self.chunks)
        print(f"Finished! Total curated chunks indexed: {len(self.chunks)}")


    def index_data(self, filepaths):
        """Load and index one or more JSON corpus files.

        Args:
            filepaths: a single path string or a list of path strings.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        # 1. Load and merge all corpora
        corpus = []
        for fp in filepaths:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            corpus.extend(data)
            print(f"Loaded {len(data):,} docs from {fp}")

        if not corpus:
            raise ValueError("All corpora are empty!")

        # 2. Chunk documents — returns list[Document]
        print(f"Indexing {len(corpus):,} documents total...")
        self.chunks = self.chunker.chunk(corpus)

        # 3. Extract text and generate embeddings
        texts = [doc.page_content for doc in self.chunks]
        embeddings = self.embedder.embed_documents(texts)

        # 4. Store Document objects + embeddings so metadata is available at retrieval time
        self.vectordb.add_documents(embeddings, self.chunks)
        print(f"Finished! Total chunks: {len(self.chunks)}")


    # Inside rag_pipeline.py
    def query(self, question, mode="socratic"):

        # Pick the LLM for this mode: 'socratic' -> socratic model, else QA model.
        llm = self.llm_socratic if mode == "socratic" else self.llm_qa

        query_embedding = self.embedder.embed_query(question)

        retrieved_docs = self.retriever.retrieve(
            question,
            query_embedding
        )

        prompt = self.prompt_builder.build_prompt(
            question,
            retrieved_docs,
            mode=mode
        )

        answer = llm.generate(prompt)

        # Guard: a small model often ignores the Socratic instruction and dumps a
        # direct answer. If that happens, retry ONCE with a stricter prompt.
        if (
            mode == "socratic"
            and hasattr(self.prompt_builder, "is_socratic_response")
            and not self.prompt_builder.is_socratic_response(answer)
        ):
            strict_prompt = self.prompt_builder.build_prompt(
                question, retrieved_docs, mode="socratic", strict=True
            )
            retry = llm.generate(strict_prompt)
            # Keep the retry only if it's actually more Socratic than the first try.
            if self.prompt_builder.is_socratic_response(retry):
                answer = retry

        return answer, retrieved_docs