import pathlib
import toml

from functools import cached_property
from pydantic import BaseModel, ConfigDict
from PIL import Image

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
    caption: str = ''

    caption_suffix: str = '.txt'
    toml_suffix: str = '.toml'
    history_suffix: str = '.history~'

    model_config = ConfigDict(extra='allow')

    @cached_property
    def absolute_path(self):
        return pathlib.Path(self.path).absolute()
    
    @cached_property
    def caption_path(self):
        return self.absolute_path.with_suffix(self.caption_suffix)
    
    @cached_property
    def toml_path(self):
        return self.absolute_path.with_suffix(self.toml_suffix)

    @cached_property
    def toml_backup_path(self):
        return self.absolute_path.with_suffix(self.toml_suffix + '~')

    @cached_property
    def history_path(self):
        return self.absolute_path.with_suffix(self.history_suffix)
    
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
        
        with open(self.caption_path, 'r') as f:
            return f.read().strip()

    def save_history(self, when_not_exists: bool = False):
        """
        Appends the current state (as TOML) to the history file.

        Args:
            when_not_exists (bool): If True, only saves history if the history file does not already exist.
        """

        if self.history_path.exists() and when_not_exists:
            return
        
        with open(self.history_path, 'a') as f:
            f.write(self._serialize_toml_history())

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

        with open(self.caption_path, 'w') as f:
            f.write(caption)

        if self.toml_path.exists():
            shutil.copy(str(self.toml_path), str(self.toml_backup_path))

        with open(self.toml_path, 'w') as f:
            f.write(self.dump_toml())

        self.caption = caption

    def _serialize_toml_history(self):
        buffer = self.dump_toml(with_caption=True).strip()
        buffer += '\n----------\n'
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
            toml_dict['caption'] = self.caption
        
        return toml.dumps(toml_dict)

