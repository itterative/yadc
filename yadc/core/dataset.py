import pathlib
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import tomlkit
from PIL import Image
from pydantic import BaseModel, ConfigDict, PrivateAttr

HISTORY_MARKER = "----------"


class DatasetImage(BaseModel):
    """
    Represents an image in a dataset, managing its file path, caption, metadata, and associated history.

    This class provides utilities to:
    - Resolve absolute paths for the image and related files.
    - Read and write captions from/to `.txt` files.
    - Save versioned metadata history using TOML format.
    - Backup metadata before updates.
    - Access the image data as a PIL.Image object (RGB mode).

    Attributes:
        path (str): Path to the image file.
        caption (str): Caption associated with the image. Defaults to empty string.
        caption_suffix (str): File extension for caption files. Default: `.txt`.
        toml_suffix (str): File extension for metadata TOML files. Default: `.toml`.
        history_suffix (str): File extension for history backup files. Default: `.history~`.

    Extra Fields:
        Any additional metadata can be stored in the model instance and will be persisted
        when saving to TOML.
    """

    path: str
    caption: str = ""

    caption_suffix: str = ".txt"
    toml_suffix: str = ".toml"
    history_suffix: str = ".history~"

    _dataset_extras: dict[str, object] = PrivateAttr(default_factory=dict)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    @cached_property
    def absolute_path(self) -> Path:
        return pathlib.Path(self.path).absolute()

    @cached_property
    def caption_path(self) -> Path:
        return self.absolute_path.with_suffix(self.caption_suffix)

    @cached_property
    def toml_path(self) -> Path:
        return self.absolute_path.with_suffix(self.toml_suffix)

    @cached_property
    def toml_backup_path(self) -> Path:
        return self.absolute_path.with_suffix(self.toml_suffix + "~")

    @cached_property
    def history_path(self) -> Path:
        return self.absolute_path.with_suffix(self.history_suffix)

    def draft_path(self, name: str) -> Path:
        """
        Returns the path for a named draft file.

        Draft files follow the naming convention: image_name.draft_name.draft~

        Args:
            name (str): The name of the draft (e.g., 'gemma', 'qwen').

        Returns:
            pathlib.Path: Path to the draft file.
        """
        return self.absolute_path.parent / (self.absolute_path.stem + "." + name + ".draft~")

    def read_draft(self, name: str):
        """
        Reads a named draft file.

        Args:
            name (str): The name of the draft to read.

        Returns:
            str: The draft content, stripped of whitespace. Empty string if not found.
        """
        path = self.draft_path(name)
        if path.exists():
            return path.read_text().strip()
        return ""

    def write_draft(self, name: str, content: str):
        """
        Writes content to a named draft file.

        Args:
            name (str): The name of the draft.
            content (str): The content to write.
        """
        self.draft_path(name).write_text(content)

    def delete_draft(self, name: str) -> bool:
        """Delete a named draft file.

        Args:
            name (str): The name of the draft to delete.

        Returns:
            bool: True if the draft was deleted, False if it didn't exist.
        """
        path = self.draft_path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def read_all_drafts(self) -> dict[str, str]:
        """
        Reads all draft files associated with this image.

        Scans for files matching the pattern: image_name.*.draft~

        Returns:
            dict[str, str]: A dictionary mapping draft names to their content.
        """
        drafts: dict[str, str] = {}
        pattern = self.absolute_path.stem + ".*.draft~"

        for path in self.absolute_path.parent.glob(pattern):
            filename = path.name
            prefix = self.absolute_path.stem + "."
            suffix = ".draft~"

            if filename.startswith(prefix) and filename.endswith(suffix):
                name = filename[len(prefix) : -len(suffix)]
                drafts[name] = path.read_text().strip()

        return drafts

    def read_image(self):
        """
        Opens and returns the image in RGB format.

        Returns:
            PIL.Image.Image: The image object converted to RGB mode.
        """

        return Image.open(self.path).convert("RGB")

    def read_caption(self):
        """
        Reads the caption from the caption file if it exists; otherwise returns the in-memory caption.

        Returns:
            str: The caption text, stripped of leading/trailing whitespace.
        """

        if not self.caption_path.exists():
            return self.caption

        with open(self.caption_path, "r") as f:
            return f.read().strip()

    def save_history(self, when_not_exists: bool = False):
        """
        Appends the current state (as TOML) to the history file.

        Args:
            when_not_exists (bool): If True, only saves history if the history file does not already exist.
        """

        if self.history_path.exists() and when_not_exists:
            return

        with open(self.history_path, "a") as f:
            f.write(self._serialize_toml_history())

    def delete_history_entry(self, index: int) -> bool:
        """Delete a history entry by index.

        Re-writes the history file without the entry at the given index.

        Args:
            index (int): 0-based index into the history list (oldest=0).

        Returns:
            bool: True if the entry was deleted, False if index was out of range.
        """
        entries = self.read_history()
        if index < 0 or index >= len(entries):
            return False

        entries.pop(index)

        # Re-write the entire history file
        if not entries:
            self.history_path.unlink(missing_ok=True)
        else:
            with open(self.history_path, "w") as f:
                for entry in entries:
                    buffer = entry.dump_toml(with_caption=True).strip()
                    f.write(buffer)
                    f.write(f"\n{HISTORY_MARKER}\n")

        return True

    def read_history(self) -> list["DatasetImage"]:
        if not self.history_path.exists():
            return []

        history: list["DatasetImage"] = []
        history_content = self.history_path.read_text().split(HISTORY_MARKER)

        for history_entry in history_content:
            history_entry = history_entry.strip()

            if not history_entry:
                continue

            try:
                history_data = tomlkit.loads(history_entry)
                history_data.setdefault("path", str(self.absolute_path))
                history.append(DatasetImage.model_validate(history_data))
            except Exception:
                continue

        return history

    def update_caption(self, caption: str):
        """
        Updates the caption by:
        - Writing it to the caption file.
        - Backing up the current TOML metadata (if it exists).
        - Saving the current model state (including extra fields) to the TOML file.
        - Updating the in-memory `caption` attribute.

        Args:
            caption (str): The new caption to set.
        """

        import shutil

        with open(self.caption_path, "w") as f:
            f.write(caption)

        if self.toml_path.exists():
            shutil.copy(str(self.toml_path), str(self.toml_backup_path))

        with open(self.toml_path, "w") as f:
            f.write(self.dump_toml())

        self.caption = caption

    def _serialize_toml_history(self):
        buffer = self.dump_toml(with_caption=True).strip()
        buffer += f"\n{HISTORY_MARKER}\n"
        return buffer

    def dump_toml(self, with_caption: bool = False):
        """
        Dumps extra model fields (metadata) to TOML format.

        Args:
            with_caption (bool): If True, includes the current `caption` in the output.

        Returns:
            str: TOML-formatted string of the metadata (and optionally caption).
        """

        toml_dict = self.__pydantic_extra__ or {}

        if with_caption:
            toml_dict["caption"] = self.caption

        return tomlkit.dumps(toml_dict)
