validator_version = "0.0.1"
version_split = validator_version.split(".")
version_int_version = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
