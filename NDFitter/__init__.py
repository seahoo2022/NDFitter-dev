"""Import backends only when used, keeping optional dependencies independent."""

from importlib import import_module

__all__ = ['utils', 'GP', 'GPyTorch', 'MLP', 'torch_interpolation', 'paths']


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
