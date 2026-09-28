"""Import backends only when used, keeping optional dependencies independent."""

from importlib import import_module

__all__ = ['train', 'train_1d', 'train_yscaled', 'train_yscaled_2by2']


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
