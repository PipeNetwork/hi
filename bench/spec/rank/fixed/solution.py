def rank(scores):
    """Competition ranking (1-based): tied scores share a rank, and the next
    distinct score skips the gap (e.g. [90,85,85,70] -> [1,2,2,4])."""
    ordered = sorted(scores, reverse=True)
    rank_of = {}
    for position, value in enumerate(ordered):
        if value not in rank_of:
            rank_of[value] = position + 1
    return [rank_of[v] for v in scores]
