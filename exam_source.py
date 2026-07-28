# Hand-transcribed from the mock-exam screenshots, then rewritten so that every
# answer is SELF-CONTAINED: it uses only information the question itself provides.
#
# WHY: the exam showed students a specific table/matrix/sentence and asked about
# it. Copying the exam's specific numbers or word-pairs into the answer (without
# the accompanying data) makes the answer look hallucinated to a user of the
# assistant, who never saw that data. So:
#   - questions with no numbers get METHOD-only answers (no invented values),
#   - where a concrete example is essential, it is moved INTO the question,
#   - "which statements are correct" style questions are reframed as open
#     conceptual questions that stand on their own.
#
# The source WEEK is NOT stored here; it is derived automatically by
# build_benchmark_from_exam.py by matching each answer against the real slides.
#
# Some entries carry "qa_gold_standard_with_example" as an OPTIONAL second answer
# variant (used where you explicitly wanted both a plain and an example version).

EXAM = [
  {
    "q": "In a dependency analysis (Universal Dependencies style, where a noun heads its prepositional phrase), if a prepositional phrase modifies a noun rather than the verb, what dependency relation connects that noun to the noun inside the prepositional phrase?",
    "answer": "It is nmod (nominal modifier): when the phrase describes the noun rather than the action, the inner noun attaches to the modified noun as a nominal modifier, not as a verbal dependent.",
    "topic": "dependency parsing", "kind": "conceptual"
  },
  {
    "q": "In a dependency parse, what is the root (head) of a simple declarative sentence, and what relation connects the subject to it?",
    "answer": "The main verb is the root of the sentence, and the subject attaches to it as a direct dependent (the subject relation). Other arguments and modifiers also hang off the verb as its dependents.",
    "topic": "dependency parsing", "kind": "conceptual"
  },
  {
    "q": "How is the observed agreement between two annotators computed when they each assign a category label to the same set of items?",
    "answer": "Observed agreement is the proportion of items on which the two annotators assign the same label: count the items they agree on and divide by the total number of items.",
    "topic": "inter-annotator agreement", "kind": "conceptual"
  },
  {
    "q": "In Cohen's Kappa, what is expected (chance) agreement and how is it computed from each annotator's label distribution?",
    "answer": "Expected agreement is the probability the two annotators would agree purely by chance. It is computed by taking, for each category, the product of the two annotators' marginal proportions for that category, and summing those products across all categories.",
    "topic": "inter-annotator agreement", "kind": "conceptual"
  },
  {
    "q": "How is Cohen's Kappa computed from observed agreement and expected agreement?",
    "answer": "Kappa = (observed agreement - expected agreement) / (1 - expected agreement). It rescales the observed agreement by removing the agreement expected by chance, so 1 means perfect agreement and 0 means chance-level agreement.",
    "topic": "inter-annotator agreement", "kind": "conceptual"
  },
  {
    "q": "What is the intuition behind inverse document frequency (IDF)?",
    "answer": "IDF down-weights terms that appear in many documents and up-weights rarer terms: the fewer documents a term occurs in, the higher its weight. This decreases the influence of common, less informative terms and increases the influence of uncommon, more discriminative ones.",
    "topic": "TF-IDF", "kind": "conceptual"
  },
  {
    "q": "Using idf(t)=log10(N/df(t)) with raw term frequency tf, how do you compute the tf-idf weight of a term in a document, and what does each part contribute?",
    "answer": "tf-idf = tf * idf(t) = tf * log10(N/df(t)), where N is the total number of documents and df(t) is the number of documents containing the term. The tf part rewards terms that occur often in the document; the idf part discounts terms that occur in many documents, so a term that is frequent in this document but rare across the collection gets the highest weight.",
    "topic": "TF-IDF", "kind": "conceptual"
  },
  {
    "q": "When computing tf-idf with idf(t)=log10(N/df(t)), why does a term that appears in more of the documents receive a smaller idf, and therefore a smaller tf-idf weight?",
    "answer": "Because df(t) grows as the term appears in more documents, the ratio N/df(t) shrinks toward 1, and log10 of a value near 1 is near 0. So a term occurring in most documents has a small idf and contributes little discriminative weight, even if its raw frequency is high.",
    "topic": "TF-IDF", "kind": "conceptual"
  },
  {
    "q": "In a term-document matrix, how is a word represented as a vector and how is a document represented as a vector?",
    "answer": "A word is represented by its row in the matrix: the counts of that word across all the documents. A document is represented by its column: the counts of every term within that document.",
    "qa_gold_standard_with_example": "A word is represented by its row in the matrix, i.e. the counts of that word across all the documents (for example, a word whose row reads 114, 80, 62, 89 has those counts in documents 1 to 4). A document is represented by its column, i.e. the counts of every term within it (for example, a document whose column reads 13, 89, 4, 3 has those term counts).",
    "topic": "vector representations", "kind": "conceptual"
  },
  {
    "q": "What are the disadvantages of using count-based vectors to represent word meanings?",
    "answer": "They are high-dimensional and sparse, which makes them hard to use effectively in machine learning, and they do not capture how a word's meaning changes across different contexts (no sense or context sensitivity).",
    "topic": "count-based vectors", "kind": "conceptual"
  },
  {
    "q": "In a named entity recognition task using the BIO tagging scheme to detect a given number of named-entity categories, how many tag classes are there in total?",
    "answer": "For k categories there are 2k + 1 tags: a B- (begin) and an I- (inside) tag for each of the k categories, plus a single O tag for tokens that are not part of any entity. (For example, 10 categories give 21 tags.)",
    "topic": "named entity recognition", "kind": "conceptual"
  },
  {
    "q": "What is Named Entity Linking (NEL), and what are the main steps involved in it?",
    "answer": "NEL maps a named-entity mention in text to the correct entry in a knowledge base, which matters for language understanding. It is typically done in two steps: candidate generation (finding plausible knowledge-base entries for the mention) followed by candidate ranking (choosing the best one). It is not trivial, and the ranking step is not restricted to neural methods.",
    "topic": "named entity linking", "kind": "conceptual"
  },
  {
    "q": "What do the WordNet relations hyponymy, hypernymy and antonymy each mean, and how do hyponymy and hypernymy relate to each other?",
    "answer": "A hyponym is a more specific term whose meaning is included in a broader term (e.g. a subtype is a hyponym of its category). A hypernym is the broader, more general term (the category). Hyponymy and hypernymy are inverses of each other: if A is a hyponym of B, then B is a hypernym of A. Antonymy relates two words with opposite meanings.",
    "topic": "WordNet / lexical semantics", "kind": "conceptual"
  },
  {
    "q": "What kind of NLP tasks can be addressed with sequence labelling models, and what property do such tasks share?",
    "answer": "Sequence labelling suits tasks that assign a label to each token in a sequence in order. Named Entity Recognition, Part-of-Speech Tagging and Semantic Role Labelling all fit this pattern. Tasks whose output is not a per-token label sequence (such as machine translation or full syntactic parsing) are not standard sequence-labelling tasks.",
    "topic": "sequence labelling", "kind": "conceptual"
  },
  {
    "q": "What factors are generally credited with the success of deep learning in NLP?",
    "answer": "The availability of large-scale text datasets together with deep-learning frameworks and specialised hardware; the ability of these models to capture the variability of natural language; and their ability to learn expressive representations from data in an unsupervised or self-supervised way.",
    "topic": "deep learning in NLP", "kind": "conceptual"
  },
  {
    "q": "In BERT's Masked Language Modelling, of the tokens chosen for masking, how are they split between the [MASK] token, a random token, and being left unchanged?",
    "answer": "Of the tokens selected for masking, 80% are replaced with the [MASK] token, 10% are replaced with a random token, and 10% are left unchanged. This mix reduces the mismatch between pre-training (which sees [MASK]) and fine-tuning (which does not).",
    "topic": "BERT / MLM", "kind": "conceptual"
  },
  {
    "q": "What are the key architectural properties of the Transformer, and how does it handle token order without recurrence?",
    "answer": "A Transformer is built from stacked layers of self-attention and feed-forward networks, and it can process all tokens in a sequence in parallel during training. Because it has no recurrence, it adds positional encodings so the model still knows the order of the tokens; self-attention then models dependencies between all tokens.",
    "topic": "transformers", "kind": "conceptual"
  },
  {
    "q": "How does the self-attention mechanism work, and which tokens can each token attend to?",
    "answer": "Self-attention derives queries, keys and values for each token; attention scores come from the similarity between queries and keys, and are used to take a weighted combination of the values. Each token can attend to every other token in the sequence (not just neighbours), and the operation keeps the sequence length unchanged.",
    "topic": "self-attention", "kind": "conceptual"
  },
  {
    "q": "What are the key properties of BERT, including how it is trained and how it uses context?",
    "answer": "BERT is trained with Masked Language Modelling and encodes each token using bidirectional context (both left and right), rather than predicting tokens strictly left to right. It uses token embeddings, and its special [CLS] token gives a summary representation that can be used for classification tasks.",
    "topic": "BERT", "kind": "conceptual"
  },
  {
    "q": "Given a table of bigram probabilities, how do you compute the probability of a sentence under a bigram model, and what tokens must be included?",
    "answer": "You multiply together the bigram probabilities of each consecutive word pair in the sentence, including the transition from the start token <s> to the first word and from the last word to the end token </s>. Each factor is looked up in the bigram table, and their product is the sentence probability.",
    "topic": "n-gram language models", "kind": "conceptual"
  },
  {
    "q": "Under a bigram model (including <s> and </s>), how do you determine which of several candidate sentences has the highest probability?",
    "answer": "For each candidate sentence, compute its probability as the product of its consecutive bigram probabilities (including the <s> and </s> transitions), then pick the sentence with the largest resulting product.",
    "topic": "n-gram language models", "kind": "conceptual"
  },
  {
    "q": "What is perplexity in language modelling, what does it measure, and does a higher or lower value indicate a better model?",
    "answer": "Perplexity measures how 'surprised' a language model is by a sequence; it is computed from the probabilities the model assigns to the tokens, and equals the exponentiated average negative log-likelihood per token. Lower perplexity means the model assigns higher probability to the data, so lower is better.",
    "topic": "perplexity", "kind": "conceptual"
  },
]
