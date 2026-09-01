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
#
# DEDUPLICATED: removed q28 (dup of q6), q32 (dup of q10), q33 (dup of q11),
# q34 (dup of q16).

EXAM = [
  {
    "q1": "In a dependency analysis (Universal Dependencies style, where a noun heads its prepositional phrase), if a prepositional phrase modifies a noun rather than the verb, what dependency relation connects that noun to the noun inside the prepositional phrase?",
    "answer": "It is nmod (nominal modifier): when the phrase describes the noun rather than the action, the inner noun attaches to the modified noun as a nominal modifier, not as a verbal dependent.",
    "topic": "dependency parsing", "kind": "conceptual"
  },
  {
    "q2": "In a dependency parse, what is the root (head) of a simple declarative sentence, and what relation connects the subject to it?",
    "answer": "The main verb is the root of the sentence, and the subject attaches to it directly as an nsubj (nominal subject) dependent. Other arguments and modifiers also hang off the verb as its dependents, so the verb is the single node from which the rest of the structure descends.",
    "topic": "dependency parsing", "kind": "conceptual"
  },
  {
    "q3": "How is the observed agreement between two annotators computed when they each assign a category label to the same set of items?",
    "answer": "Observed agreement is the proportion of items on which the two annotators assign the same label: count the items they agree on and divide by the total number of items.",
    "topic": "inter-annotator agreement", "kind": "conceptual"
  },
  {
    "q4": "In Cohen's Kappa, what is expected (chance) agreement and how is it computed from each annotator's label distribution?",
    "answer": "Expected agreement is the probability the two annotators would agree purely by chance. It is computed by taking, for each category, the product of the two annotators' marginal proportions for that category, and summing those products across all categories.",
    "topic": "inter-annotator agreement", "kind": "conceptual"
  },
  {
    "q5": "How is Cohen's Kappa computed from observed agreement and expected agreement?",
    "answer": "Kappa = (observed agreement - expected agreement) / (1 - expected agreement). It rescales the observed agreement by removing the agreement expected by chance, so 1 means perfect agreement and 0 means chance-level agreement.",
    "topic": "inter-annotator agreement", "kind": "conceptual"
  },
  {
    "q6": "What problem does inverse document frequency (IDF) solve in document representation, and what kind of term does it favour?",
    "answer": "Raw term frequency alone rewards whatever occurs most often, which is dominated by function words and other terms that appear across the whole collection and therefore say nothing about what distinguishes one document from another. IDF fixes this by weighting each term according to how few documents it appears in, so terms concentrated in a small number of documents carry more weight than terms spread across all of them. The effect is that a document's representation is driven by its discriminative vocabulary rather than by its most common words.",
    "topic": "TF-IDF", "kind": "conceptual"
  },
  {
    "q7": "Using idf(t)=log10(N/df(t)) with raw term frequency tf, how do you compute the tf-idf weight of a term in a document, and what does each part contribute?",
    "answer": "tf-idf = tf * idf(t) = tf * log10(N/df(t)), where N is the total number of documents and df(t) is the number of documents containing the term. The tf part rewards terms that occur often in the document; the idf part discounts terms that occur in many documents, so a term that is frequent in this document but rare across the collection gets the highest weight.",
    "topic": "TF-IDF", "kind": "conceptual"
  },
  {
    "q8": "When computing tf-idf with idf(t)=log10(N/df(t)), why does a term that appears in more of the documents receive a smaller idf, and therefore a smaller tf-idf weight?",
    "answer": "Because df(t) grows as the term appears in more documents, the ratio N/df(t) shrinks toward 1, and log10 of a value near 1 is near 0. So a term occurring in most documents has a small idf and contributes little discriminative weight, even if its raw frequency is high.",
    "topic": "TF-IDF", "kind": "conceptual"
  },
  # note Q9 down
  {
    "q9": "In a term-document matrix, how is a word represented as a vector and how is a document represented as a vector?",
    "answer": "A word is represented by its row in the matrix: the counts of that word across all the documents. A document is represented by its column: the counts of every term within that document.",
    "topic": "vector representations", "kind": "conceptual"
  },
    # note Q10 down, there's a mismatch.
  {
    "q10": "What are the disadvantages of using count-based vectors to represent word meanings?",
    "answer": "They are high-dimensional and sparse, which makes them hard to use effectively in machine learning, and they do not capture how a word's meaning changes across different contexts (no sense or context sensitivity).",
    "topic": "count-based vectors", "kind": "conceptual"
  },
  {
    "q11": "In a named entity recognition task using the BIO tagging scheme to detect a given number of named-entity categories, how many tag classes are there in total?",
    "answer": "For k categories there are 2k + 1 tags: a B- (begin) and an I- (inside) tag for each of the k categories, plus a single O tag for tokens that are not part of any entity. (For example, 10 categories give 21 tags.)",
    "topic": "named entity recognition", "kind": "conceptual"
  },
  {
    "q12": "What is Named Entity Linking (NEL), and what are the main steps involved in it?",
    "answer": "NEL maps a named-entity mention in text to the correct entry in a knowledge base, which matters for language understanding. It is typically done in two steps: candidate generation (finding plausible knowledge-base entries for the mention) followed by candidate ranking (choosing the best one).",
    "topic": "named entity linking", "kind": "conceptual"
  },
  {
    "q13": "What do the WordNet relations hyponymy, hypernymy and antonymy each mean, and how do hyponymy and hypernymy relate to each other?",
    "answer": "A hyponym is a more specific term whose meaning is included in a broader term (e.g. a subtype is a hyponym of its category). A hypernym is the broader, more general term (the category). Hyponymy and hypernymy are inverses of each other: if A is a hyponym of B, then B is a hypernym of A. Antonymy relates two words with opposite meanings.",
    "topic": "WordNet / lexical semantics", "kind": "conceptual"
  },
  {
    "q14": "What kind of NLP tasks can be addressed with sequence labelling models, and what property do such tasks share?",
    "answer": "Sequence labelling suits tasks that assign a label to each token in a sequence in order. Named Entity Recognition, Part-of-Speech Tagging and Semantic Role Labelling all fit this pattern.",
    "topic": "sequence labelling", "kind": "conceptual"
  },

  {
    "q15": "In BERT's Masked Language Modelling, of the tokens chosen for masking, how are they split between the [MASK] token, a random token, and being left unchanged?",
    "answer": "Of the tokens selected for masking, 80% are replaced with the [MASK] token, 10% are replaced with a random token, and 10% are left unchanged.",
    "topic": "BERT / MLM", "kind": "conceptual"
  },
  {
    "q16": "What are the key architectural properties of the Transformer, and how does it handle token order without recurrence?",
    "answer": "A Transformer is built from stacked layers of self-attention and feed-forward networks, and it can process all tokens in a sequence in parallel during training. Because it has no recurrence, it adds positional encodings so the model still knows the order of the tokens; self-attention then models dependencies between all tokens.",
    "topic": "transformers", "kind": "conceptual"
  },
  {
    "q17": "How does the self-attention mechanism work, and which tokens can each token attend to?",
    "answer": "Self-attention derives queries, keys and values for each token; attention scores come from the similarity between queries and keys, and are used to take a weighted combination of the values. Each token can attend to every other token in the sequence (not just neighbours), and the operation keeps the sequence length unchanged.",
    "topic": "self-attention", "kind": "conceptual"
  },
  {
    "q18": "What are the key properties of BERT, including how it is trained and how it uses context?",
    "answer": "BERT is trained with Masked Language Modelling and encodes each token using bidirectional context (both left and right), rather than predicting tokens strictly left to right. It uses token embeddings, and its special [CLS] token gives a summary representation that can be used for classification tasks.",
    "topic": "BERT", "kind": "conceptual"
  },
  {
    "q19": "Given a table of bigram probabilities, how do you compute the probability of a sentence under a bigram model, and what tokens must be included?",
    "answer": "You multiply together the bigram probabilities of each consecutive word pair in the sentence, including the transition from the start token <s> to the first word and from the last word to the end token </s>.",
    "topic": "n-gram language models", "kind": "conceptual"
  },
  {
    "q20": "Under a bigram model (including <s> and </s>), how do you determine which of several candidate sentences has the highest probability?",
    "answer": "For each candidate sentence, compute its probability as the product of its consecutive bigram probabilities (including the <s> and </s> transitions), then pick the sentence with the largest resulting product.",
    "topic": "n-gram language models", "kind": "conceptual"
  },
  {
    "q21": "What is perplexity in language modelling, what does it measure, and does a higher or lower value indicate a better model?",
    "answer": "Perplexity measures how 'surprised' a language model is by a sequence; it is computed from the probabilities the model assigns to the tokens, and equals the exponentiated average negative log-likelihood per token. Lower perplexity means the model assigns higher probability to the data, so lower is better.",
    "topic": "perplexity", "kind": "conceptual"
  },
  {
"q22": "Consider the sentence 'The boy saw the man with a telescope.' Assume a dependency analysis in which nouns are the heads of prepositional phrases, as in Universal Dependencies. Under the interpretation 'the boy saw the man who had a telescope', which dependency relation holds between the tokens 'man' and 'telescope'?",
"answer": "The relation is nmod: 'telescope' is a nominal modifier of 'man', because in this reading the prepositional phrase 'with a telescope' attaches to the noun 'man' rather than to the verb.",
"topic": "dependency parsing",
"kind": "conceptual"
},
{
"q23": "Consider the sentence 'The boy saw the man with a telescope.' interpreted as meaning that the boy saw the man who had a telescope. In the dependency graph for this interpretation, which tokens are dependents of the head of the sentence?",
"answer": "The head of the sentence is the verb 'saw', and its dependents are 'boy' (subject), 'man' (object) and the full stop '.' (punctuation). 'telescope' is not a dependent of 'saw' because it hangs off 'man' in this reading.",
"topic": "dependency parsing",
"kind": "conceptual"
},
{
"q24": "Two annotators independently label four social media posts as POSITIVE, NEGATIVE or NEUTRAL. Annotator 1 gives POSITIVE, NEGATIVE, NEUTRAL, POSITIVE. Annotator 2 gives POSITIVE, NEGATIVE, POSITIVE, POSITIVE. What is the observed agreement used in the Kappa coefficient?",
"answer": "The observed agreement is 0.75, because the two annotators assign the same label on 3 of the 4 posts (they differ only on the third post, NEUTRAL vs POSITIVE), and 3/4 = 0.75.",
"topic": "inter-annotator agreement",
"kind": "calculation"
},
{
"q25": "Two annotators label four posts. Annotator 1 gives POSITIVE, NEGATIVE, NEUTRAL, POSITIVE. Annotator 2 gives POSITIVE, NEGATIVE, POSITIVE, POSITIVE. What is the expected (chance) agreement used in the Kappa coefficient?",
"answer": "The expected agreement is 0.438. Annotator 1 uses POSITIVE 2/4, NEGATIVE 1/4, NEUTRAL 1/4; Annotator 2 uses POSITIVE 3/4, NEGATIVE 1/4, NEUTRAL 0. Summing the products per category gives (0.5 x 0.75) + (0.25 x 0.25) + (0.25 x 0) = 0.4375, which rounds to 0.438.",
"topic": "inter-annotator agreement",
"kind": "calculation"
},
{
"q26": "For a set of four annotated posts, the observed agreement between two annotators is 0.75 and the expected agreement is 0.438. What is the value of the Kappa coefficient?",
"answer": "Kappa is about 0.555. Using Kappa = (observed - expected) / (1 - expected), this is (0.75 - 0.4375) / (1 - 0.4375) = 0.3125 / 0.5625 = 0.555.",
"topic": "inter-annotator agreement",
"kind": "calculation"
},
{
"q27": "Given four preprocessed documents - Doc1: 'my cat meow all the time while my dog does n't bark .', Doc2: 'can cat be friendly ?', Doc3: 'i do n't know why my dog always bark at squirrel .', Doc4: 'dog be sad when being home alone .' - and using idf(t) = log10(N/df(t)) with raw term frequency tf, what is tf-idf('bark', Doc1) to three decimal places?",
"answer": "tf-idf('bark', Doc1) = 0.301. The term 'bark' occurs once in Doc1 (tf = 1) and appears in 2 of the 4 documents (Doc1 and Doc3), so idf = log10(4/2) = 0.301, and 1 x 0.301 = 0.301.",
"topic": "tf-idf",
"kind": "calculation"
},
{
"q28": "Given four preprocessed documents - Doc1: 'my cat meow all the time while my dog does n't bark .', Doc2: 'can cat be friendly ?', Doc3: 'i do n't know why my dog always bark at squirrel .', Doc4: 'dog be sad when being home alone .' - and using idf(t) = log10(N/df(t)) with raw term frequency tf, what is tf-idf('dog', Doc3) to three decimal places?",
"answer": "tf-idf('dog', Doc3) = 0.125. The term 'dog' occurs once in Doc3 (tf = 1) and appears in 3 of the 4 documents (Doc1, Doc3, Doc4), so idf = log10(4/3) = 0.125, and 1 x 0.125 = 0.125.",
"topic": "tf-idf",
"kind": "calculation"
},

{
"q29": "In a term-document matrix built from 4 documents, the term 'battle' occurs 7 times in 'Julius Caesar' and appears in 3 of the 4 documents (As You Like It, Julius Caesar and Henry V, but not Twelfth Night). Using idf(t) = log10(N/df(t)) and raw term frequency tf, what is the tf-idf value of 'battle' in 'Julius Caesar', rounded to 3 decimal places?",
"answer": "tf-idf('battle', 'Julius Caesar') = 0.874. With tf = 7 and df = 3 out of N = 4 documents, idf = log10(4/3) = 0.125, so 7 x 0.125 = 0.874.",
"topic": "tf-idf",
"kind": "calculation"
},
{
  "q30": "A bigram model gives the following probabilities: P(arctic | <s>) = 0.6, P(monkeys | arctic) = 0.1, P(are | monkeys) = 0.417, P(my | are) = 0.15, P(favourite | my) = 0.1, P(band | favourite) = 0.1, P(</s> | band) = 0.9. Using this model, what is the probability of the sentence 'arctic monkeys are my favourite band', including the start token <s> and end token </s>, rounded to 5 decimal places?",
  "answer": "The probability is 0.00003. It is computed as the product of the seven bigram probabilities along the sentence: 0.6 * 0.1 * 0.417 * 0.15 * 0.1 * 0.1 * 0.9 = 0.000033777, which rounds to 0.00003 to 5 decimal places.",
  "topic": "n-gram language models",
  "kind": "calculation"
}
]