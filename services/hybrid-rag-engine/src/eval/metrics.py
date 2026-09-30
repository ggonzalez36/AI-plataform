import re
from typing import List, Dict, Any
from src.api.models import EvaluationMetrics

class RAGEvaluator:
    @staticmethod
    def evaluate(query: str, retrieved_snippets: List[str], synthesized_answer: str) -> EvaluationMetrics:
        """
        Calculates production RAG evaluation metrics:
        1. Faithfulness: Is the answer grounded in the retrieved context?
        2. Answer Relevancy: Does the answer address the user query?
        3. Context Precision: Are the most relevant chunks ranked at top positions?
        """
        if not retrieved_snippets:
            return EvaluationMetrics(faithfulness=0.0, answer_relevancy=0.0, context_precision=0.0)

        # 1. Faithfulness (Grounding)
        answer_tokens = set(re.findall(r"\b\w{3,}\b", synthesized_answer.lower()))
        combined_context = " ".join(retrieved_snippets).lower()
        context_tokens = set(re.findall(r"\b\w{3,}\b", combined_context))

        if answer_tokens:
            grounded_tokens = answer_tokens.intersection(context_tokens)
            faithfulness = min(1.0, max(0.1, len(grounded_tokens) / len(answer_tokens)))
        else:
            faithfulness = 0.5

        # 2. Answer Relevancy
        query_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if query_tokens:
            addressed_tokens = query_tokens.intersection(answer_tokens)
            relevancy = min(1.0, max(0.15, len(addressed_tokens) / len(query_tokens) + 0.3))
        else:
            relevancy = 0.8

        # 3. Context Precision (Position-weighted relevance)
        precision_scores = []
        for rank, snippet in enumerate(retrieved_snippets):
            snippet_tokens = set(re.findall(r"\b\w{3,}\b", snippet.lower()))
            overlap = query_tokens.intersection(snippet_tokens)
            precision_at_k = (len(overlap) / len(query_tokens)) if query_tokens else 0.5
            rank_weight = 1.0 / (rank + 1)
            precision_scores.append(precision_at_k * rank_weight)

        context_precision = min(1.0, max(0.1, sum(precision_scores) / (sum(1.0 / (r + 1) for r in range(len(retrieved_snippets))) or 1.0)))

        return EvaluationMetrics(
            faithfulness=round(faithfulness, 3),
            answer_relevancy=round(relevancy, 3),
            context_precision=round(context_precision, 3),
        )

evaluator = RAGEvaluator()
