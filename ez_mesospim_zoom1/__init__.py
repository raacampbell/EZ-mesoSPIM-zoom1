from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("EZ-mesoSPIM-zoom1")
except PackageNotFoundError:
    # package is not installed
    pass
