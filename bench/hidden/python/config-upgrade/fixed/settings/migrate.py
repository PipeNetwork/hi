from copy import deepcopy


def migrate_v1(config):
    migrated = deepcopy(config)
    host = migrated.pop("host")
    port = migrated.pop("port")
    debug = migrated.get("debug", False)
    if isinstance(debug, str):
        debug = debug.strip().lower() in {"yes", "true", "1", "on"}
    migrated["debug"] = bool(debug)
    migrated["server"] = {"host": host, "port": port}
    migrated["schema_version"] = 2
    return migrated
