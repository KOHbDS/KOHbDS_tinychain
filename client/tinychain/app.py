import inspect

from . import chain, graph
from .collection import Column, table
from .decorators import MethodStub
from .ref import Get, MethodSubject
from .reflect.op import Op
from .state import Instance, State
from .util import to_json, uri
from .value import U32


class Model(object):
    """A model defining the attributes of a class used by an `App`."""

    @staticmethod
    def schema():
        return []

class AppChain(chain.Block):
    # TODO: implement backup and versioning support for App using AppChain
    pass


def model(model_config):
    if not inspect.isclass(model_config):
        raise ValueError(f"model configuration must be a class, not {model_config}")

    parents = model_config.mro()
    if len(parents) <= 1 or not issubclass(parents[1], State):
        raise ValueError(f"a user-defined class must inherit from a native class")

    schema = model_config.schema()
    attributes = {}
    for field in schema:
        if field.name in attributes:
            raise ValueError(f"{model_config} has a duplicate entry for field {field.name}")
        elif not issubclass(field.dtype, State):
            raise ValueError(f"field data type must be a TinyChain State, not {field.dtype}")

        def getter(self):
            return field.dtype(Get(MethodSubject(self, field.name)))

        attributes[field.name] = property(getter)

    return type(model_config.__name__, model_config.__bases__, attributes)


class App(object):
    @staticmethod
    def _export():
        return []

    def __init__(self):
        schema = graph.Schema()
        reserved = set(["graph"])

        for cls in self._export():
            if not inspect.isclass(cls) or not issubclass(cls, Instance):
                raise RuntimeError(f"App can only export a TinyChain class, not {cls}")
            elif cls.__name__ in reserved:
                raise ValueError(f"App.{cls.__name__} is a reserved name")

            key = [Column(cls.__name__.lower() + "_id", U32)]
            values = []
            for field in cls.schema():
                if isinstance(field, Column):
                    values.append(field)
                else:
                    raise NotImplementedError("App does not yet support graph edges")

            table_schema = table.Schema(key, values)
            schema.create_table(cls.__name__.lower() + "s", table_schema)

        class AppGraph(graph.Graph):
            __uri__ = uri(self) + "/graph"

            @classmethod
            def create(cls):
                return cls(schema)

        self.graph = AppChain(AppGraph.create())

    def __json__(self):
        if not hasattr(self, "__uri__"):
            raise RuntimeError(f"{self} has no URI")

        attributes = {}
        attributes["graph"] = to_json(self.graph)

        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue

            if isinstance(attr, Op):
                raise NotImplementedError("App does not yet support methods")

            attributes[name] = to_json(attr)

        for cls in self._export():
            attributes[cls.__name__] = to_json(cls)

        return {str(uri(self)): attributes}
