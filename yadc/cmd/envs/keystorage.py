"""``KeyStorage`` ABC for loading/saving the RSA private key (implemented by keyring and password backends)."""

import abc


class KeyStorage(abc.ABC):
    """Abstract backend for persisting the RSA private key."""

    @abc.abstractmethod
    def load_private_key(self) -> bytes | None:
        """Return the raw PEM-encoded private key, or ``None`` if not found."""

    @abc.abstractmethod
    def save_private_key(self, private_key_pem: bytes) -> None:
        """Persist the raw PEM-encoded private key."""

    @abc.abstractmethod
    def delete_private_key(self) -> None:
        """Remove the persisted private key."""

    @abc.abstractmethod
    def has_private_key(self) -> bool:
        """Return ``True`` if a private key is stored."""
