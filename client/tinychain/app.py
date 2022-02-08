"""A hosted :class:`App` or :class:`Library`."""

import inspect
import logging

from tinychain.chain import Chain
from tinychain.decorators import MethodStub
from tinychain.reflect import Meta
from tinychain.state import Scalar, State
from tinychain.util import get_ref, to_json, uri, URI


class App(object):
    def __json__(self):
        form = {}

        for cls in self.exports():
            if not inspect.isclass(cls):
                raise ValueError(f"App can only export a class, not {cls}")
            elif type(cls) != Meta:
                logging.warning(f"{cls} is not of type {Meta} and may not support JSON encoding")

            form[cls.__name__] = to_json(cls)

        # TODO: de-duplicate header-builder code from reflect.meta
        class Header(cls):
            pass

        header = Header(uri(self))
        instance = cls(uri(self))

        for name, attr in inspect.getmembers(self):
            if name.startswith('_') or isinstance(attr, URI):
                continue
            elif inspect.ismethod(attr) and attr.__self__ is cls:
                # it's a @classmethod
                continue

            if isinstance(attr, MethodStub):
                setattr(header, name, attr.method(instance, name))
            elif isinstance(attr, State):
                member_uri = uri(self).append(name)
                attr_ref = get_ref(attr, member_uri)

                if not uri(attr_ref) == member_uri:
                    raise RuntimeError(f"failed to assign URI {member_uri} to instance attribute {attr_ref} "
                                       + f"(assigned URI is {uri(attr_ref)})")

                setattr(header, name, attr_ref)
            else:
                setattr(header, name, attr)

        for name, attr in inspect.getmembers(self):
            if name.startswith('_') or name == "exports":
                continue

            if is_mutable(attr) and not isinstance(attr, Chain):
                raise RuntimeError("mutable state must be in a Chain")
            elif isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(header, name))
            else:
                form[name] = to_json(attr)

        return {str(uri(self)): form}


class Library(object):
    @staticmethod
    def exports():
        return []

    def __json__(self):
        form = {}

        for cls in self.exports():
            if not inspect.isclass(cls):
                raise ValueError(f"Library can only export a class, not {cls}")
            elif type(cls) != Meta:
                logging.warning(f"{cls} is not of type {Meta} and may not support JSON encoding")

            form[cls.__name__] = to_json(cls)

        for name, attr in inspect.getmembers(self):
            if name.startswith('_') or name == "exports":
                continue

            if is_mutable(attr):
                raise RuntimeError("a Library may not contain mutable state")
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

        raise RuntimeError(f"There is already an entry at {config_path}")
    else:
        import os

        if not config_path.parent.exists():
            os.makedirs(config_path.parent)

        with open(config_path, 'w') as config_file:
            config_file.write(json.dumps(config, indent=4))
