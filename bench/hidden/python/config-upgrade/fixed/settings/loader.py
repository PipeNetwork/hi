import json

from .migrate import migrate_v1


def load(path):
    with open(path) as handle:
        config = json.load(handle)
    if config.get("schema_version", 1) == 1:
        return migrate_v1(config)
    return config
