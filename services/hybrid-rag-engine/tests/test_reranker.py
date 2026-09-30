from src.core.reranker import reranker

def test_reciprocal_rank_fusion_combined_ranking():
    dense_ranked = [
        {"id": "doc-1", "score": 0.9, "title": "Doc 1", "snippet": "Text 1"},
        {"id": "doc-2", "score": 0.7, "title": "Doc 2", "snippet": "Text 2"},
    ]
    sparse_ranked = [
        {"id": "doc-2", "score": 10.5, "title": "Doc 2", "snippet": "Text 2"},
        {"id": "doc-3", "score": 8.0, "title": "Doc 3", "snippet": "Text 3"},
    ]

    fused = reranker.reciprocal_rank_fusion(
        dense_ranked=dense_ranked,
        sparse_ranked=sparse_ranked,
        dense_weight=0.5,
        sparse_weight=0.5,
        top_k=3,
    )

    assert len(fused) <= 3
    # doc-2 is present in both lists, so its combined RRF should place it high
    top_doc_ids = [d["id"] for d in fused]
    assert "doc-2" in top_doc_ids
    assert fused[0]["score"] > 0.0

def test_cross_encoder_rerank_keyword_boost():
    candidates = [
        {"id": "doc-a", "score": 0.5, "title": "General Policies", "snippet": "Nothing about encryption here"},
        {"id": "doc-b", "score": 0.5, "title": "Security Standard", "snippet": "AES-256 encryption is required for all data"},
    ]
    query = "AES-256 encryption standard"
    reranked = reranker.cross_encoder_rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["id"] == "doc-b"
    assert reranked[0]["score"] > reranked[1]["score"]
