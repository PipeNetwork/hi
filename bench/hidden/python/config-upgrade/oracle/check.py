import json
import tempfile

from settings import load, migrate_v1

original = {
    "schema_version": 1,
    "host": "127.0.0.1",
    "port": 8080,
    "debug": "YeS",
    "metadata": {"owner": "ops"},
}
migrated = migrate_v1(original)
assert original["schema_version"] == 1 and "host" in original
assert migrated == {
    "schema_version": 2,
    "server": {"host": "127.0.0.1", "port": 8080},
    "debug": True,
    "metadata": {"owner": "ops"},
}
with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
    json.dump(original, handle)
    handle.flush()
    assert load(handle.name) == migrated
v2 = {"schema_version": 2, "server": {"host": "x", "port": 1}, "debug": False}
with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
    json.dump(v2, handle)
    handle.flush()
    assert load(handle.name) == v2
