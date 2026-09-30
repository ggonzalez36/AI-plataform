import math
from src.core.embeddings import embedding_engine

def test_dense_embedding_dimension_and_norm():
    text = "Compliance audit logs must be retained for seven years."
    vec = embedding_engine.generate_dense(text)
    
    assert len(vec) == 384
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3

def test_dense_embedding_deterministic():
    text = "Deterministic vector generation test."
    vec1 = embedding_engine.generate_dense(text)
    vec2 = embedding_engine.generate_dense(text)
    
    assert vec1 == vec2

def test_sparse_embedding_generation():
    text = "API Gateway rate limiting SOX compliance."
    indices, values = embedding_engine.generate_sparse(text)
    
    assert len(indices) > 0
    assert len(values) == len(indices)
    assert all(isinstance(i, int) and i > 0 for i in indices)
    assert all(isinstance(v, float) and v > 0.0 for v in values)
