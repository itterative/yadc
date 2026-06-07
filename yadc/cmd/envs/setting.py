"""``Setting`` base class and ``EncryptionMethod`` enum (``none`` / ``keyring`` / ``password``)."""

import enum

from typing_extensions import override


class EncryptionMethod(enum.StrEnum):
    NONE = "none"
    KEYRING = "keyring"
    PASSWORD = "password"


def parse_prefixed_value(value: str | None) -> tuple[str | None, EncryptionMethod]:
    """Extract ciphertext and encryption method from a stored value.

    Backward compatibility: values without a prefix are assumed to use keyring.
    """
    if not value:
        return None, EncryptionMethod.NONE
    if value.startswith("keyring:"):
        return value[8:], EncryptionMethod.KEYRING
    if value.startswith("password:"):
        return value[9:], EncryptionMethod.PASSWORD
    # No prefix: legacy keyring-encrypted value
    return value, EncryptionMethod.KEYRING


class Setting:
    value: str | None
    method: EncryptionMethod

    def __init__(self, value: str | None, method: EncryptionMethod = EncryptionMethod.NONE):
        self.value = value
        self.method = method

    @override
    def __str__(self) -> str:
        if not self.value:
            return ""

        if self.method != EncryptionMethod.NONE:
            return "[REDACTED]"

        return self.value
