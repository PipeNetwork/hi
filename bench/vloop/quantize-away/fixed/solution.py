def quantize(value, step):
    """Round to nearest multiple of step; ties away from zero."""
    sign = -1 if value < 0 else 1
    return sign * ((abs(value) + step // 2) // step * step)
