from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cl")
except PackageNotFoundError:
    __version__ = "unknown"
