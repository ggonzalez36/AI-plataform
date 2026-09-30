import math
import re
import hashlib
from typing import List, Tuple, Dict

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

class HybridEmbeddingEngine:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._fastembed_model = None
        self._fastembed_available = False

        try:
            from fastembed import TextEmbedding
            self._fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._fastembed_available = True
        except Exception:
            # Graceful deterministic fallback without external network/heavy binaries
            self._fastembed_available = False

    def generate_dense(self, text: str) -> List[float]:
        """Generates a normalized dense vector of dimension self.dim"""
        if self._fastembed_available and self._fastembed_model:
            try:
                embeddings = list(self._fastembed_model.embed([text]))
                if len(embeddings) > 0:
                    vec = embeddings[0].tolist()
                    return vec[:self.dim]
            except Exception:
                pass

        # High-performance deterministic subword hashing embedding
        vec = [0.0] * self.dim
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            if token in STOP_WORDS:
                continue
            # Token feature projection
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) == 1 else -1.0
            vec[idx] += sign * 1.5

            # Subword 3-grams for morphological capture
            for i in range(len(token) - 2):
                gram = token[i:i+3]
                gh = int(hashlib.sha256(gram.encode("utf-8")).hexdigest(), 16)
                g_idx = gh % self.dim
                g_sign = 1.0 if ((gh >> 4) & 1) == 1 else -1.0
                vec[g_idx] += g_sign * 0.5

        # L2 Normalization (Unit vector for Cosine Similarity)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [round(x / norm, 6) for x in vec]
        return vec

    def generate_sparse(self, text: str) -> Tuple[List[int], List[float]]:
        """Generates BM25-like term frequency sparse vectors (indices and weights)"""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return [], []

        tf_map: Dict[str, int] = {}
        for token in tokens:
            if token not in STOP_WORDS and len(token) > 1:
                tf_map[token] = tf_map.get(token, 0) + 1

        if not tf_map:
            return [], []

        indices: List[int] = []
        values: List[float] = []

        doc_len = len(tokens)
        avg_len = 25.0
        k1 = 1.2
        b = 0.75

        for term, tf in tf_map.items():
            h = int(hashlib.sha256(term.encode("utf-8")).hexdigest(), 16)
            token_id = (h % 1000000) + 1 # Positive sparse feature ID

            # BM25 term saturation score
            tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avg_len)))
            indices.append(token_id)
            values.append(round(tf_norm, 4))

        return indices, values

# Singleton instance
embedding_engine = HybridEmbeddingEngine()
