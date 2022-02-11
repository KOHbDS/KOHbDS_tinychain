"""A hosted :class:`App` or :class:`Library`."""

import inspect
import logging

from tinychain.chain import Chain
from tinychain.decorators import MethodStub
from tinychain.reflect import header, Meta
from tinychain.state import Instance, Scalar, State
from tinychain.util import get_ref, to_json, uri, URI


class Model(object, metaclass=Meta):
    def __ref__(self, name):
        raise RuntimeError("cannot reference a Model itself; try using App.<model name>.load")

    def _is_static(self):
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue

            if issubclass(attr, State):
                return False

        return True


def model(cls):
    if not issubclass(cls, Model):
        raise ValueError(f"Model must be a class, not {cls}")

    if not hasattr(cls, "__uri__"):
        raise ValueError(f"Model {cls} must have a URI (hint: set the __uri__ attribute of the class)")

    try:
        proto = cls()
    except TypeError as e:
        raise TypeError(f"error constructing model for {cls}: {e}")

    attrs = {}
    for name, attr in inspect.getmembers(proto):
        if name.startswith('_'):
            continue
        elif isinstance(attr, MethodStub):
            continue
        elif hasattr(Model, name):
            continue

        if not inspect.isclass(attr):
            raise TypeError(f"model attribute must be a type of State, not {attr}")

        if issubclass(attr, Model):
            attr = model(attr)
        elif not issubclass(attr, State):
            raise TypeError(f"unknown model attribute type: {attr}")

        attrs[name] = attr(URI("self").append(name))

    class _Model(Instance, cls):
        __uri__ = cls.__uri__

        def __init__(self, form):
            Instance.__init__(self, form)

            for name in attrs:
                setattr(self, name, attrs[name])

        def __ref__(self, name):
            return self.__class__(URI(name))

    _Model.__name__ = cls.__name__
    return _Model


class Library(object):
    @staticmethod
    def exports():
        return []

    def __init__(self, uri=None):
        if uri is not None:
            self.__uri__ = uri

        for cls in self.exports():
            setattr(self, cls.__name__, model(cls))

        self._allow_mutable = False

    def __json__(self):
        form = handle_exports(self)

        for name, attr in inspect.getmembers(self):
            if name.startswith('_') or name == "exports":
                continue

            _, instance_header = header(type(self))

            if not self._allow_mutable and is_mutable(attr):
                raise RuntimeError(f"{self.__class__.__name__} may not contain mutable state")
            if self._allow_mutable and is_mutable(attr) and not isinstance(attr, Chain):
                raise RuntimeError("mutable state must be in a Chain")
            elif isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(instance_header, name))
            else:
                form[name] = to_json(attr)

        return {str(uri(self)): form}


class App(Library):
    def __init__(self, uri=None):
        Library.__init__(self, uri)
        self._allow_mutable = True


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
        if not issubclass(cls, Model):
            raise ValueError(f"Library can only export a Model class, not {cls}")
        elif type(cls) != Meta:
            logging.warning(f"{cls} is not of type {Meta} and may not support JSON encoding")

        expected_uri = uri(app_or_lib).append(cls.__name__)

        if uri(cls) == expected_uri:
            pass
        else:
            raise ValueError(f"the URI of {cls} should be {expected_uri}, not {uri(cls)}")

        form[cls.__name__] = to_json(model(cls))

    return form
