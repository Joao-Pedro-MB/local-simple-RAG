"""Reciprocal rank fusion (Cormack, Clarke and Büttcher, 2009): merges rankings by rank alone.

    fused score of a Chunk = sum, over the rankings that hold it, of 1 / (60 + its rank)

Only ranks are used, never scores, so rankings on unrelated scales merge without any
normalizing: BM25 scores are unbounded, while cosine similarities lie between -1 and 1. The
constant 60 flattens the curve. First place earns 1/61 and tenth place 1/70, so one ranking's
favourite cannot outvote everything the other ranking says. A ranking that does not hold a
Chunk adds nothing for it.
"""

from collections import defaultdict

RRF_CONSTANT = 60


def reciprocal_rank_fusion(rankings):
    """Merge rankings of Index positions, each best first, into (position, fused score) pairs.

    The result is best first. Equal fused scores keep Index order, so the result never
    depends on the order of a dict or a set.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            fused[position] += 1 / (RRF_CONSTANT + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
