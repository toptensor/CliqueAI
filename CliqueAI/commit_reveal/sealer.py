import hashlib
import hmac
import secrets
from dataclasses import dataclass

from CliqueAI.graph.codec import DEFAULT_BASE92_ALPHABET

KEY_BYTES = 32

_ALPHABET = DEFAULT_BASE92_ALPHABET
_BASE = len(_ALPHABET)
_INDEX = {char: index for index, char in enumerate(_ALPHABET)}
# Reject the byte remainder to avoid modulo bias.
_SAMPLE_LIMIT = (256 // _BASE) * _BASE


class SealError(ValueError):
    """Raised when a sealed challenge cannot be verified."""


@dataclass(frozen=True)
class SealedChallenge:
    payload: str
    digest: str
    key: str


def _validate_uuid(uuid: str) -> None:
    if not uuid or len(uuid) > 64:
        raise SealError("uuid must contain between 1 and 64 characters")


def _length_prefixed(value: str) -> bytes:
    # Keep adjacent variable-length values unambiguous.
    data = value.encode("utf-8")
    return len(data).to_bytes(4, "big") + data


def commitment(uuid: str, payload: str) -> str:
    """Return the graph commitment for a challenge."""
    return hashlib.sha256(_length_prefixed(uuid) + payload.encode("utf-8")).hexdigest()


def keystream(key: bytes, uuid: str, length: int) -> list[int]:
    """Derive unbiased base92 digits."""
    seed = key + _length_prefixed(uuid)
    size = length * 2 + 64
    while True:
        stream = hashlib.shake_256(seed).digest(size)
        digits = [byte % _BASE for byte in stream if byte < _SAMPLE_LIMIT]
        if len(digits) >= length:
            return digits[:length]
        size *= 2


def _shift(payload: str, pads: list[int], direction: int) -> str:
    try:
        return "".join(
            _ALPHABET[(_INDEX[char] + direction * pad) % _BASE]
            for char, pad in zip(payload, pads)
        )
    except KeyError as error:
        raise SealError(f"character outside the base92 alphabet: {error}") from error


def seal(uuid: str, payload: str, *, key: bytes | None = None) -> SealedChallenge:
    """Seal a base92 payload without changing its length."""
    _validate_uuid(uuid)
    key = key if key is not None else secrets.token_bytes(KEY_BYTES)
    if len(key) != KEY_BYTES:
        raise SealError(f"key must be {KEY_BYTES} bytes")
    if any(char not in _INDEX for char in payload):
        raise SealError("payload contains characters outside the base92 alphabet")
    return SealedChallenge(
        payload=_shift(payload, keystream(key, uuid, len(payload)), 1),
        digest=commitment(uuid, payload),
        key=key.hex(),
    )


def unseal(*, uuid: str, payload: str, key: str, digest: str) -> str:
    """Recover and verify a sealed payload."""
    _validate_uuid(uuid)
    try:
        key_bytes = bytes.fromhex(key)
    except ValueError as error:
        raise SealError(f"key must be hex encoded: {error}") from error
    if len(key_bytes) != KEY_BYTES:
        raise SealError(f"key must be {KEY_BYTES} bytes")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SealError("digest must be 64 lowercase hex characters")

    plaintext = _shift(payload, keystream(key_bytes, uuid, len(payload)), -1)
    if not hmac.compare_digest(commitment(uuid, plaintext), digest):
        raise SealError("digest mismatch: the revealed key does not open the commit")
    return plaintext
