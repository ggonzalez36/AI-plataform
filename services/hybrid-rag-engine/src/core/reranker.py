import math
import re
from typing import List, Dict, Any

class HybridReranker:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self,
        dense_ranked: List[Dict[str, Any]],
        sparse_ranked: List[Dict[str, Any]],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Fuses ranked candidate lists from Dense and Sparse searches using Reciprocal Rank Fusion (RRF).
        RRF_score(d) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)
        """
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        dense_scores_map: Dict[str, float] = {}
        sparse_scores_map: Dict[str, float] = {}

        # 1. Process Dense Ranking
        for rank, item in enumerate(dense_ranked):
            doc_id = str(item["id"])
            doc_map[doc_id] = item
            dense_scores_map[doc_id] = float(item.get("score", 0.0))
            score = dense_weight / (self.rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        # 2. Process Sparse Ranking
        for rank, item in enumerate(sparse_ranked):
            doc_id = str(item["id"])
            if doc_id not in doc_map:
                doc_map[doc_id] = item
            sparse_scores_map[doc_id] = float(item.get("score", 0.0))
            score = sparse_weight / (self.rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        # 3. Sort by combined RRF score descending
        sorted_ids = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)

        max_score = max(doc_scores.values()) if doc_scores else 1.0
        results: List[Dict[str, Any]] = []

        for doc_id in sorted_ids[:top_k]:
            base_item = dict(doc_map[doc_id])
            raw_rrf = doc_scores[doc_id]
            norm_score = round(raw_rrf / max_score if max_score > 0 else raw_rrf, 4)

            base_item["rrf_score"] = norm_score
            base_item["dense_score"] = dense_scores_map.get(doc_id)
            base_item["sparse_score"] = sparse_scores_map.get(doc_id)
            base_item["score"] = norm_score
            results.append(base_item)

        return results

    def cross_encoder_rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Cross-Encoder simulation: refines relevance using query-term coverage,
        exact phrase alignment, and token density.
        """
        if not candidates:
            return []

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        if not query_terms:
            return candidates[:top_k]

        reranked = []
        for c in candidates:
            snippet = c.get("snippet", "") + " " + c.get("title", "")
            snippet_tokens = re.findall(r"\b\w+\b", snippet.lower())
            snippet_set = set(snippet_tokens)

            overlap = query_terms.intersection(snippet_set)
            term_coverage = len(overlap) / len(query_terms) if query_terms else 0.0

            # Exact substring match bonus
            exact_bonus = 0.2 if query.lower() in snippet.lower() else 0.0

            # Cross-score fusion
            initial_score = c.get("score", 0.5)
            cross_score = (initial_score * 0.6) + (term_coverage * 0.3) + exact_bonus
            cross_score = min(0.99, max(0.01, cross_score))

            c_copy = dict(c)
            c_copy["score"] = round(cross_score, 4)
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]

# Singleton instance
reranker = HybridReranker()
