def rotate_left(values, amount):
    if not values:
        return []
    amount %= len(values)
    return list(values[amount:]) + list(values[:amount])
