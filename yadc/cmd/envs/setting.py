from typing import Optional

class Setting:
    def __init__(self, value: Optional[str], encrypted: bool = False):
        self.value = value
        self.encrypted = encrypted

    def __str__(self) -> str:
        if not self.value:
            return ''

        if self.encrypted:
            return '[REDACTED]'

        return self.value
