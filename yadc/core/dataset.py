import pathlib
import toml

from functools import cached_property
from pydantic import BaseModel, ConfigDict
from PIL import Image

class DatasetImage(BaseModel):
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
        return Image.open(self.path).convert("RGB")

    def read_caption(self):
        if not self.caption_path.exists():
            return self.caption
        
        with open(self.caption_path, 'r') as f:
            return f.read().strip()

    def save_history(self, when_not_exists: bool = False):
        if self.history_path.exists() and when_not_exists:
            return
        
        with open(self.history_path, 'a') as f:
            f.write(self._serialize_toml_history())

    def update_caption(self, caption: str):
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
        toml_dict = self.__pydantic_extra__ or {}

        if with_caption:
            toml_dict['caption'] = self.caption
        
        return toml.dumps(toml_dict)

