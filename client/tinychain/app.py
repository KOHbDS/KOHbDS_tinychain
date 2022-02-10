"""A hosted :class:`App` or :class:`Library`."""

import inspect
import logging

from tinychain.chain import Chain
from tinychain.decorators import MethodStub
from tinychain.reflect import header, Meta
from tinychain.state import Instance, Scalar, State
from tinychain.util import get_ref, to_json, uri, URI


def model(cls):
    if not inspect.isclass(cls):
        raise ValueError(f"Model must be a class, not {cls}")

    if not hasattr(cls, "__uri__"):
        raise ValueError(f"a Model must have a URI (hint: set the __uri__ attribute of the class)")

    if issubclass(cls, Instance):
        return cls
    else:
        class Model(Instance, cls, metaclass=Meta):
            __uri__ = cls.__uri__

            def __init__(self, form=None):
                Instance.__init__(self, form)

        Model.__name__ = cls.__name__
        return Model


class App(object):
    def __json__(self):
        form = handle_exports(self.exports())

        _, instance_header = header(self, False)

        for name, attr in inspect.getmembers(self):
            if name.startswith('_') or name == "exports":
                continue

            if is_mutable(attr) and not isinstance(attr, Chain):
                raise RuntimeError("mutable state must be in a Chain")
            elif isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(instance_header, name))
            else:
                form[name] = to_json(attr)

        return {str(uri(self)): form}


class Library(object):
    @staticmethod
    def exports():
        return []

    def __json__(self):
        form = handle_exports(self)

        for name, attr in inspect.getmembers(self):
            if name.startswith('_') or name == "exports":
                continue

            _, instance_header = header(type(self), False)

            if is_mutable(attr):
                raise RuntimeError("a Library may not contain mutable state")
            elif isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(instance_header, name))
            else:
                form[name] = to_json(attr)

        return {str(uri(self)): form}


def is_mutable(state):
    if not isinstance(state, State):
        return False

    if isinstance(state, Scalar):
        return False,

    return True


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

        raise RuntimeError(f"there is already an entry at {config_path}")
    else:
        import os

        if not config_path.parent.exists():
            os.makedirs(config_path.parent)

        with open(config_path, 'w') as config_file:
            config_file.write(json.dumps(config, indent=4))


def handle_exports(app_or_lib):
    form = {}

    for cls in app_or_lib.exports():
        if not inspect.isclass(cls):
            raise ValueError(f"Library can only export a class, not {cls}")
        elif type(cls) != Meta:
            logging.warning(f"{cls} is not of type {Meta} and may not support JSON encoding")

        expected_uri = uri(app_or_lib).append(cls.__name__)

        if uri(cls) == expected_uri:
            pass
        else:
            raise ValueError(f"the URI of {cls} should be {expected_uri}, not {uri(cls)}")

        form[cls.__name__] = to_json(cls)

    return form
