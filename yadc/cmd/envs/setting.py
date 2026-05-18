from typing_extensions import override


class Setting:
    value: str | None
    encrypted: bool

    def __init__(self, value: str | None, encrypted: bool = False):
        self.value = value
        self.encrypted = encrypted

    @override
    def __str__(self) -> str:
        if not self.value:
            return ""

        if self.encrypted:
            return "[REDACTED]"

        return self.value
