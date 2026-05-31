"""Dataset resolution: scanning paths, merging inline images, applying extras."""

import pathlib
from typing import Callable

import toml

from .config import ConfigDatasetEntry
from .dataset import DatasetImage

# Type for a function that reads a DatasetImage from disk.
# Returns None if the file is not a valid image.
ReadImageFn = Callable[[str, str], DatasetImage | None]


def read_image_from_disk(file_path: str, caption_suffix: str) -> DatasetImage | None:
    """Read and parse a single dataset image from disk.

    Opens the image to validate it, loads any associated TOML metadata,
    and reads the existing caption if present.

    Args:
        file_path: Path to the image file.
        caption_suffix: File extension for caption files (e.g. '.txt').

    Returns:
        A populated DatasetImage, or None if the file is not a valid image.
    """
    try:
        dataset_image = DatasetImage(path=file_path)
        dataset_image.read_image()
    except Exception:
        return None

    if not dataset_image.toml_path.exists():
        dataset_image_toml = {}
    else:
        try:
            with open(dataset_image.toml_path, "r") as f:
                dataset_image_toml = toml.load(f)
        except Exception:
            return None

    dataset_image_toml["path"] = str(dataset_image.absolute_path)
    dataset_image_toml["caption_suffix"] = caption_suffix

    image = DatasetImage.model_validate(dataset_image_toml)
    image.caption = image.read_caption()
    return image


def apply_extras_defaults(dataset_image: DatasetImage, extras: dict[str, object]):
    """Apply dataset-level extras as defaults on a DatasetImage.

    Per-image extras take priority: dataset extras are only set for keys
    that are not already present in the image's extra fields.
    """
    if not extras:
        return

    if dataset_image.__pydantic_extra__ is None:
        dataset_image.__pydantic_extra__ = {}

    for k, v in extras.items():
        if k not in dataset_image.__pydantic_extra__:
            dataset_image.__pydantic_extra__[k] = v


def reapply_dataset_extras(dataset_image: DatasetImage):
    """Re-apply dataset-level extras after a DatasetImage has been reconstructed.

    Uses the ``_dataset_extras`` dict stored during resolution. If no
    dataset extras were recorded, this is a no-op.
    """
    apply_extras_defaults(dataset_image, dataset_image._dataset_extras)


def resolve_dataset(
    entries: list[ConfigDatasetEntry],
    caption_suffix: str,
    base_dir: str | None = None,
    read_image: ReadImageFn = read_image_from_disk,
) -> list[DatasetImage]:
    """Resolve all dataset entries into a flat list of DatasetImages.

    For each entry:
    - Scans the path directory for images (if set).
    - Merges inline images with scanned images, with inline extras taking priority.
    - Applies dataset-level extras as defaults (per-image extras override).

    Args:
        entries: The dataset entries from the parsed config.
        caption_suffix: File extension for caption files.
        base_dir: Directory to resolve relative paths against. If None,
            relative paths are treated as-is (from the current working directory).
        read_image: Function used to read an image from disk. Defaults to
            read_image_from_disk. Can be overridden for testing.

    Returns:
        A flat list of resolved DatasetImages with all extras merged.
    """
    dataset: list[DatasetImage] = []
    i_dataset: dict[pathlib.Path, DatasetImage] = {}

    for entry in entries:
        # scan path directory for images
        if entry.path:
            path = pathlib.Path(entry.path)
            if not path.is_absolute() and base_dir is not None:
                path = pathlib.Path(base_dir) / path

            if path.is_dir():
                for file_path in path.iterdir():
                    dataset_image = read_image(str(file_path), caption_suffix)

                    if dataset_image is None:
                        continue

                    apply_extras_defaults(dataset_image, entry.extras)
                    dataset_image._dataset_extras = entry.extras

                    dataset.append(dataset_image)
                    i_dataset[dataset_image.absolute_path] = dataset_image

        # merge inline images with scanned images
        for dataset_image in entry.images:
            existing = i_dataset.get(dataset_image.absolute_path)

            if existing is None:
                existing = read_image(str(dataset_image.absolute_path), caption_suffix)

            if existing is not None:
                i_dataset[existing.absolute_path] = existing
                dataset.append(existing)
            else:
                apply_extras_defaults(dataset_image, entry.extras)
                dataset_image._dataset_extras = entry.extras

                i_dataset[dataset_image.absolute_path] = dataset_image
                dataset.append(dataset_image)
                continue

            # merge extras from inline image onto existing
            if existing.__pydantic_extra__ is None:
                existing.__pydantic_extra__ = {}
            if dataset_image.__pydantic_extra__ is None:
                dataset_image.__pydantic_extra__ = {}

            for k, v in dataset_image.__pydantic_extra__.items():
                if v:
                    existing.__pydantic_extra__[k] = v

            existing.caption = dataset_image.caption or existing.caption
            existing._dataset_extras = entry.extras

    return dataset
