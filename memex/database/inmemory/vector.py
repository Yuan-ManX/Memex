from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime
from typing import List, Tuple, cast

import numpy as np


# ============================================================
# Core Similarity
# ============================================================


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: Vector A
        b: Vector B

    Returns:
        Cosine similarity score
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


# ============================================================
# Salience Scoring
# ============================================================


def compute_salience_score(
    *,
    similarity: float,
    reinforcement_count: int,
    last_reinforced_at: datetime | None,
    recency_decay_days: float = 30.0,
) -> float:
    """
    Compute salience-aware score.

    Combines:

        similarity
        reinforcement strength
        recency decay

    Formula:

        similarity
        * log(reinforcement + 1)
        * recency_decay

    Recency uses half-life decay.

    Args:
        similarity:
            Cosine similarity between query and memory

        reinforcement_count:
            Number of times memory was reinforced

        last_reinforced_at:
            Last reinforcement timestamp

        recency_decay_days:
            Half-life decay window

    Returns:
        Salience score
    """

    # Reinforcement factor
    reinforcement_factor = math.log(reinforcement_count + 1)

    # Recency factor
    if last_reinforced_at is None:
        recency_factor = 0.5
    else:
        now = (
            datetime.now(last_reinforced_at.tzinfo)
            if last_reinforced_at.tzinfo
            else datetime.utcnow()
        )

        days_ago = (now - last_reinforced_at).total_seconds() / 86400

        # Half-life exponential decay
        recency_factor = math.exp(-0.693 * days_ago / recency_decay_days)

    return similarity * reinforcement_factor * recency_factor


# ============================================================
# Vector Search
# ============================================================


def topk_cosine(
    query_vec: List[float],
    corpus: Iterable[Tuple[str, List[float] | None]],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k items using cosine similarity.

    Args:
        query_vec:
            Query embedding vector

        corpus:
            Iterable of (id, embedding)

        k:
            Number of results to return

    Returns:
        List of (id, similarity_score)
    """

    ids: List[str] = []
    vecs: List[List[float]] = []

    for _id, vec in corpus:
        if vec is None:
            continue
        ids.append(_id)
        vecs.append(cast(List[float], vec))

    if not vecs:
        return []

    q = np.array(query_vec, dtype=np.float32)
    matrix = np.array(vecs, dtype=np.float32)

    q_norm = np.linalg.norm(q)
    vec_norms = np.linalg.norm(matrix, axis=1)

    scores = matrix @ q / (vec_norms * q_norm + 1e-9)

    n = len(scores)
    actual_k = min(k, n)

    if actual_k == n:
        indices = np.argsort(scores)[::-1]
    else:
        indices = np.argpartition(scores, -actual_k)[-actual_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

    return [(ids[i], float(scores[i])) for i in indices]


# ============================================================
# Salience-aware Retrieval
# ============================================================


def topk_salience(
    query_vec: List[float],
    corpus: Iterable[Tuple[str, List[float] | None, int, datetime | None]],
    *,
    k: int = 5,
    recency_decay_days: float = 30.0,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k memories using salience-aware scoring.

    Ranking formula:

        similarity
        * log(reinforcement + 1)
        * recency_decay

    Args:
        query_vec:
            Query embedding vector

        corpus:
            Iterable of:
                (id, embedding, reinforcement_count, last_reinforced_at)

        k:
            Number of results

        recency_decay_days:
            Recency half-life window

    Returns:
        Sorted list of (memory_id, salience_score)
    """

    q = np.array(query_vec, dtype=np.float32)

    scored: List[Tuple[str, float]] = []

    for _id, vec, reinforcement_count, last_reinforced_at in corpus:
        if vec is None:
            continue

        v = np.array(cast(List[float], vec), dtype=np.float32)

        similarity = cosine_similarity(q, v)

        score = compute_salience_score(
            similarity=similarity,
            reinforcement_count=reinforcement_count,
            last_reinforced_at=last_reinforced_at,
            recency_decay_days=recency_decay_days,
        )

        scored.append((_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:k]


# ============================================================
# Utility Query
# ============================================================


def query_cosine(
    query_vec: List[float],
    vectors: List[List[float]],
) -> List[Tuple[int, float]]:
    """
    Compute cosine similarity between query and vector list.

    Returns indices with similarity scores.

    Args:
        query_vec:
            Query embedding

        vectors:
            List of vectors

    Returns:
        List of (index, similarity)
    """

    q = np.array(query_vec, dtype=np.float32)

    results: List[Tuple[int, float]] = []

    for i, vec in enumerate(vectors):
        v = np.array(vec, dtype=np.float32)
        score = cosine_similarity(q, v)
        results.append((i, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return results


__all__ = [
    "cosine_similarity",
    "compute_salience_score",
    "topk_cosine",
    "topk_salience",
    "query_cosine",
]
