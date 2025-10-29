"""
DOTA-v2 dataset class names configuration.

This module provides the class names for the DOTA-v2 dataset in multiple formats
for use across different parts of the application.
"""

# List of class names (ordered by class ID)
CLASS_NAMES_LIST: list[str] = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
    "container crane",
    "airport",
    "helipad",
]


# Dictionary mapping class ID to class name
CLASS_NAMES_DICT: dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES_LIST)}


def get_class_names_list() -> list[str]:
    """Return class names as a list."""
    return CLASS_NAMES_LIST.copy()


def get_class_names_dict() -> dict[int, str]:
    """Return class names as a dictionary mapping ID to name."""
    return CLASS_NAMES_DICT.copy()


# Backward compatibility: maintain the same function name used in predict.py
def load_class_names() -> dict[int, str]:
    """Load class names from the DOTA-v2 dataset configuration."""
    return get_class_names_dict()
