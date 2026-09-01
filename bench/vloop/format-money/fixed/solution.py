def fmt(cents):
    """Integer cents -> '$1,234.56' (negatives as '-$1,234.56')."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"
