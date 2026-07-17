def total(items, coupon):
    """Receipt total in cents; coupon rules per the task."""
    if coupon not in (None, "PCT10", "OFF500"):
        raise ValueError(f"unknown coupon: {coupon!r}")
    subtotal = sum(q * p for q, p in items)
    if coupon == "PCT10":
        discount = (subtotal * 10 + 50) // 100
        return subtotal - discount
    if coupon == "OFF500":
        return max(0, subtotal - 500)
    return subtotal
