import math


def round_to_int(x):
    """Round float x to the nearest integer, ties away from zero."""
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))
