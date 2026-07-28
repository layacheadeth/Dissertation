import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _to_inputs(enc, device):
    """Normalise the output of tokenizer.apply_chat_template.

    Depending on the transformers version, apply_chat_template(..., return_tensors="pt")
    may return either a raw input_ids tensor OR a BatchEncoding/dict. This helper makes
    downstream code (model.generate(**inputs), inputs["input_ids"]) work in both cases.
    """
    if hasattr(enc, "input_ids") and hasattr(enc, "items"):   # BatchEncoding
        enc = {k: v for k, v in enc.items()}
    if isinstance(enc, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    # raw tensor
    return {"input_ids": enc.to(device)}


class PromptTemplate:
    def __init__(self):
        # Positive framing works far better than a wall of "never" rules on small models.
        self.socratic_system = """You are EduBot, a Socratic tutor for the university course COMP64702.

Your goal is NOT to answer. Your goal is to reply with ONE short guiding question that
nudges the student to work the answer out for themselves, using ONLY the CONTEXT.

Rules you must follow every time:
- Reply with a single probing question (1-2 sentences). End with a question mark.
- Never state the answer, a definition, or a fact outright. Never confirm or deny.
- No lists, no bullet points, no numbered steps, no headings, no bold text.
- Anchor your question to a specific idea named in the CONTEXT, but do not explain it.
- If the CONTEXT does not cover the topic, ask which part of the material they want to explore."""

        self.qa_system = """You are EduBot, a direct academic assistant for the course COMP64702.
Rules: Answer the question directly, factually, and concisely using ONLY the provided context. Do not use outside knowledge."""

        # Few-shot demonstrations of GOOD Socratic replies, in the course domain.
        # Small models need to *see* the target behaviour, not just be told about it.
        self.socratic_few_shot = """Study these examples of the required style, then do the CURRENT TASK.

EXAMPLE 1
CONTEXT: Cosine similarity measures the angle between two vectors and normalises by their magnitudes, unlike the raw dot product which grows with vector length.
QUESTION: Why can the raw dot product be misleading when comparing a long document to a short one?
SOCRATIC REPLY: If one document were simply ten times longer but about the same topic, what would happen to the magnitude of its vector, and how does dividing by that magnitude change the comparison?

EXAMPLE 2
CONTEXT: TF-IDF weights a term by how often it occurs in a document, and downweights terms that occur in many documents via inverse document frequency (IDF), which uses a logarithm.
QUESTION: How does TF-IDF treat a word that appears in every document?
SOCRATIC REPLY: If a term's document frequency equalled the total number of documents, what value would the IDF fraction reduce to, and what is the logarithm of that value?

EXAMPLE 3
CONTEXT: One-hot vectors assign each token a value of 1 at a unique index and 0 everywhere else, so distinct tokens occupy orthogonal dimensions.
QUESTION: Why do 'apricot' and 'pineapple' show no relationship as one-hot vectors?
SOCRATIC REPLY: If each word's single 1 sits at a different index from every other word, what do you get when you take the dot product of two vectors that share no non-zero position?"""

    def build_prompt(self, question, contexts, mode="socratic", strict=False):
        context_text = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else doc for doc in contexts
        )

        # Route to the requested system persona based on the classifier choice
        system_prompt = self.socratic_system if mode == "socratic" else self.qa_system

        if mode == "socratic":
            reminder = ""
            if strict:
                # Second-attempt reminder when the first reply broke character.
                reminder = ("\n\nIMPORTANT: Your previous style was wrong. Do NOT explain or answer. "
                            "Output ONLY one guiding question ending in '?'. No lists, no facts.")
            user_content = (
                f"{self.socratic_few_shot}\n\n"
                f"---\nCURRENT TASK\nCONTEXT:\n{context_text}\n\n"
                f"QUESTION: {question}\n\nSOCRATIC REPLY:{reminder}"
            )
        else:
            user_content = f"CONTEXT:\n{context_text}\n\nQUESTION: {question}\n\nANSWER:"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def is_socratic_response(text):
        """Cheap heuristic to catch a reply that slipped into direct-answer mode."""
        t = (text or "").strip()
        if "?" not in t:
            return False
        # Reject bullet / numbered lists, headings and bold — hallmarks of a QA dump.
        if re.search(r'(^|\n)\s*([-*•]|\d+[.)])\s', t):
            return False
        if "###" in t or "**" in t:
            return False
        # A genuine Socratic reply is short; a paragraph-long lecture is not.
        if len(t.split()) > 90:
            return False
        return True


class QwenLLM:

    def __init__(self, device="cpu", model_name=MODEL_NAME):
        print(f"Loading LLM ({model_name})...")
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device in ("mps", "cuda") else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, messages, max_tokens=512):
        enc = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = _to_inputs(enc, self.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )
        return response.strip()
