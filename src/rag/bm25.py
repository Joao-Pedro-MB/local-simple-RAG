"""BM25 keyword search over Chunk text, written out by hand.

Embeddings match meaning; BM25 matches the exact words. It finds the Chunks that contain the
Query's words, favouring words that few Chunks contain.

Tokenization, the same for Chunks and Queries:
  - lowercase the text;
  - a word is a run of letters and digits, or several such runs joined by @ . - _ or / with no
    space between: pass@k, top-k, bge-small-en-v1.5, raw/notes;
  - a joined word yields itself and then each of its parts, so "pass@k" gives pass@k, pass, k.
    The whole word lets exact jargon match exactly; its parts let "top k" still find "top-k".
There is no stemming and no stop-word list, because IDF already makes a word found in most
Chunks count for little.

Okapi BM25 scores a Chunk D by summing, over each distinct Query term t that D contains,

    idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * |D| / avgdl))

where f is how often t occurs in D, |D| is D's length in tokens and avgdl is the average length.
IDF uses the form Lucene and Elasticsearch use, which never goes negative:

    idf(t) = ln(1 + (N - n + 0.5) / (n + 0.5))      N Chunks, n of which contain t

The original Robertson–Spärck Jones IDF drops the "1 +" and turns negative once a term is in
more than half the Chunks, which would penalize a Chunk for containing a Query word.
"""

import math
import re
from collections import Counter

# Elasticsearch's defaults (rank_bm25 uses k1 = 1.5; the literature suggests 1.2 to 2.0).
K1 = 1.2  # saturation: each repeat of a word adds less, and the total never exceeds k1 + 1
B = 0.75  # length normalization: 0 ignores length, 1 scales fully by length relative to average

PART = re.compile(r"[^\W_]+")  # a run of letters and digits (\w without the underscore)
WORD = re.compile(r"[^\W_]+(?:[@.\-_/][^\W_]+)*")  # parts joined by @ . - _ / with no space


def tokenize(text):
    """Lowercased words, where a joined word also yields its parts: "pass@k" -> pass@k, pass, k."""
    tokens = []
    for word in WORD.findall(text.lower()):
        parts = PART.findall(word)
        if len(parts) > 1:
            tokens.append(word)
        tokens.extend(parts)
    return tokens


class BM25:
    """Corpus statistics for a list of Chunk texts, and the BM25 ranking of a Query against them."""

    def __init__(self, chunk_texts):
        self.term_counts = [Counter(tokenize(text)) for text in chunk_texts]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(self.lengths)
        # How many Chunks contain each term (not how often it occurs overall).
        self.chunks_containing = Counter(term for counts in self.term_counts for term in counts)

    def rank(self, query):
        """(position, score) for every Chunk containing a Query term, best first.

        Each distinct Query term counts once. A stable sort keeps ties in Chunk order.
        """
        terms = list(dict.fromkeys(tokenize(query)))
        scores = [(position, self._score(terms, position)) for position in range(len(self.term_counts))]
        matches = [(position, score) for position, score in scores if score > 0]
        return sorted(matches, key=lambda match: -match[1])

    def _score(self, terms, position):
        counts, length = self.term_counts[position], self.lengths[position]
        score = 0.0
        for term in terms:
            f = counts[term]
            if f:
                score += self._idf(term) * f * (K1 + 1) / (f + K1 * (1 - B + B * length / self.average_length))
        return score

    def _idf(self, term):
        n, total = self.chunks_containing[term], len(self.term_counts)
        return math.log(1 + (total - n + 0.5) / (n + 0.5))
