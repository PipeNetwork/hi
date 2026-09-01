def badge(name, role, years):
    """Badge label: 'ROLE: Name (suffix)' — see task rules."""
    role = role.upper() if role else "STAFF"
    name = name.title()
    if len(name) > 12:
        name = name[:11] + "~"
    if years == 0:
        suffix = "(new)"
    elif years >= 10:
        suffix = "(veteran)"
    else:
        suffix = f"({years}y)"
    return f"{role}: {name} {suffix}"
