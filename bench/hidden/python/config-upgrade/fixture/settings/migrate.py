def migrate_v1(config):
    config["schema_version"] = 2
    return config
