base_version = "0.0.3"
validator_version = "0.0.3"


def _version_to_int(version_str: str) -> int:
    version_split = version_str.split(".")
    major = int(version_split[0])
    minor = int(version_split[1])
    patch = int(version_split[2])
    return (10000 * major) + (100 * minor) + patch


base_int_version = _version_to_int(base_version)
validator_int_version = _version_to_int(validator_version)
