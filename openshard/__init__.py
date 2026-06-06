from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openshard")
except PackageNotFoundError:
    __version__ = "0.3.0-dev"
