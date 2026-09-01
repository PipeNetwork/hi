import json


def load(path):
    with open(path) as handle:
        return json.load(handle)
