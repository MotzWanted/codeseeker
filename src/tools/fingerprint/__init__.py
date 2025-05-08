from datasets.fingerprint import Hasher


def fingerprint(value: object) -> str:
    """Fingerprint a value."""
    return Hasher().hash(value)
