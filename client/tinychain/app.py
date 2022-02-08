"""A hosted :class:`App` or :class:`Library`."""

import inspect
import logging

from tinychain.util import to_json, uri


class App(object):
    @staticmethod
    def exports(self):
        []


class Library(object):
    @staticmethod
    def exports():
        return []

    def __json__(self):
        exports = {}
        for cls in self.exports():
            if not inspect.isclass(cls):
                raise ValueError(f"Library can only export a class, not {cls}")

            exports[cls.__name__] = to_json(cls)

        return {str(uri(self)): exports}


def write_config(app_or_library, config_path, overwrite=False):
    """Write the configuration of the given :class:`tc.App` or :class:`Library` to the given path."""

    if inspect.isclass(app_or_library):
        raise ValueError(f"write_app expects an instance of App, not a class: {app_or_library}")

    import json
    import pathlib

    config = to_json(app_or_library)
    config_path = pathlib.Path(config_path)
    if config_path.exists() and not overwrite:
        with open(config_path) as f:
            try:
                if json.load(f) == config:
                    return
            except json.decoder.JSONDecodeError as e:
                logging.warning(f"invalid JSON at {config_path}: {e}")

        raise RuntimeError(f"There is already an entry at {config_path}")
    else:
        import os

        if not config_path.parent.exists():
            os.makedirs(config_path.parent)

        with open(config_path, 'w') as config_file:
            config_file.write(json.dumps(config, indent=4))
