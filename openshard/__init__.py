from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openshard")
except PackageNotFoundError:
    __version__ = "0.2.0-dev"
