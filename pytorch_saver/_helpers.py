import json
import os


def _remove_saved_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass


def _is_json_serializable(obj) -> bool:
    """check if an object is JSON-serializable."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
