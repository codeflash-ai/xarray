from __future__ import annotations

import functools
import inspect
import itertools
import sys
import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable

from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint
from xarray.core.utils import module_available

if TYPE_CHECKING:
    import os
    from importlib.metadata import EntryPoint

    if sys.version_info >= (3, 10):
        from importlib.metadata import EntryPoints
    else:
        EntryPoints = list[EntryPoint]
    from io import BufferedIOBase

    from xarray.backends.common import AbstractDataStore

STANDARD_BACKENDS_ORDER = ["netcdf4", "h5netcdf", "scipy"]


def remove_duplicates(entrypoints: EntryPoints) -> list[EntryPoint]:  # type: ignore
    # Optimize deduplication using a dict to group by name, keeping first and emitting warning for dups
    # This avoids sort/groupby/set
    seen = {}
    duplicates = {}
    for ep in entrypoints:
        name = ep.name
        if name not in seen:
            seen[name] = ep
        else:
            if name not in duplicates:
                duplicates[name] = [seen[name], ep]
            else:
                duplicates[name].append(ep)
    # Warn about duplicates
    for name, matches in duplicates.items():
        matches_len = len(matches)
        all_module_names = [e.value.split(":")[0] for e in matches]
        selected_module_name = all_module_names[0]
        warnings.warn(
            f"Found {matches_len} entrypoints for the engine name {name}:"
            f"\n {all_module_names}.\n "
            f"The entrypoint {selected_module_name} will be used.",
            RuntimeWarning,
        )
    return list(seen.values())


def detect_parameters(open_dataset: Callable) -> tuple[str, ...]:
    signature = inspect.signature(open_dataset)
    parameters = signature.parameters
    parameters_list = []
    for name, param in parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise TypeError(
                f"All the parameters in {open_dataset!r} signature should be explicit. "
                "*args and **kwargs is not supported"
            )
        if name != "self":
            parameters_list.append(name)
    return tuple(parameters_list)


def backends_dict_from_pkg(
    entrypoints: list[EntryPoint],
) -> dict[str, type[BackendEntrypoint]]:
    backend_entrypoints = {}
    for entrypoint in entrypoints:
        name = entrypoint.name
        try:
            backend = entrypoint.load()
        except Exception as ex:
            warnings.warn(f"Engine {name!r} loading failed:\n{ex}", RuntimeWarning)
            continue
        backend_entrypoints[name] = backend
    return backend_entrypoints


def set_missing_parameters(
    backend_entrypoints: dict[str, type[BackendEntrypoint]],
) -> None:
    # Optimize: cache detect_parameters per unique backend class
    computed_params = {}
    for backend in backend_entrypoints.values():
        if backend.open_dataset_parameters is None:
            if backend in computed_params:
                backend.open_dataset_parameters = computed_params[backend]
            else:
                open_dataset = backend.open_dataset
                params = detect_parameters(open_dataset)
                backend.open_dataset_parameters = params
                computed_params[backend] = params


def sort_backends(
    backend_entrypoints: dict[str, type[BackendEntrypoint]],
) -> dict[str, type[BackendEntrypoint]]:
    # Optimize: combine pop and set operation, then extend
    ordered_backends_entrypoints = {
        be_name: backend_entrypoints.pop(be_name)
        for be_name in STANDARD_BACKENDS_ORDER
        if be_name in backend_entrypoints
    }
    if backend_entrypoints:
        ordered_backends_entrypoints.update(
            {name: backend_entrypoints[name] for name in sorted(backend_entrypoints)}
        )
    return ordered_backends_entrypoints


def build_engines(entrypoints: EntryPoints) -> dict[str, BackendEntrypoint]:  # type: ignore
    backend_entrypoints: dict[str, type[BackendEntrypoint]] = {}
    for backend_name, (module_name, backend) in BACKEND_ENTRYPOINTS.items():
        if module_name is None or module_available(module_name):
            backend_entrypoints[backend_name] = backend
    entrypoints_unique = remove_duplicates(entrypoints)
    external_backend_entrypoints = backends_dict_from_pkg(entrypoints_unique)
    backend_entrypoints.update(external_backend_entrypoints)
    backend_entrypoints = sort_backends(backend_entrypoints)
    set_missing_parameters(backend_entrypoints)
    return {name: backend() for name, backend in backend_entrypoints.items()}


@functools.lru_cache(maxsize=1)
def list_engines() -> dict[str, BackendEntrypoint]:
    """
    Return a dictionary of available engines and their BackendEntrypoint objects.

    Returns
    -------
    dictionary

    Notes
    -----
    This function lives in the backends namespace (``engs=xr.backends.list_engines()``).
    If available, more information is available about each backend via ``engs["eng_name"]``.

    # New selection mechanism introduced with Python 3.10. See GH6514.
    """
    if sys.version_info >= (3, 10):
        entrypoints = entry_points(group="xarray.backends")
    else:
        entrypoints = entry_points().get("xarray.backends", [])
    return build_engines(entrypoints)


def refresh_engines() -> None:
    """Refreshes the backend engines based on installed packages."""
    list_engines.cache_clear()


def guess_engine(
    store_spec: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
) -> str | type[BackendEntrypoint]:
    engines = list_engines()

    for engine, backend in engines.items():
        try:
            if backend.guess_can_open(store_spec):
                return engine
        except PermissionError:
            raise
        except Exception:
            warnings.warn(f"{engine!r} fails while guessing", RuntimeWarning)

    compatible_engines = []
    for engine, (_, backend_cls) in BACKEND_ENTRYPOINTS.items():
        try:
            backend = backend_cls()
            if backend.guess_can_open(store_spec):
                compatible_engines.append(engine)
        except Exception:
            warnings.warn(f"{engine!r} fails while guessing", RuntimeWarning)

    installed_engines = [k for k in engines if k != "store"]
    if not compatible_engines:
        if installed_engines:
            error_msg = (
                "did not find a match in any of xarray's currently installed IO "
                f"backends {installed_engines}. Consider explicitly selecting one of the "
                "installed engines via the ``engine`` parameter, or installing "
                "additional IO dependencies, see:\n"
                "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html\n"
                "https://docs.xarray.dev/en/stable/user-guide/io.html"
            )
        else:
            error_msg = (
                "xarray is unable to open this file because it has no currently "
                "installed IO backends. Xarray's read/write support requires "
                "installing optional IO dependencies, see:\n"
                "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html\n"
                "https://docs.xarray.dev/en/stable/user-guide/io"
            )
    else:
        error_msg = (
            "found the following matches with the input file in xarray's IO "
            f"backends: {compatible_engines}. But their dependencies may not be installed, see:\n"
            "https://docs.xarray.dev/en/stable/user-guide/io.html \n"
            "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html"
        )

    raise ValueError(error_msg)


def get_backend(engine: str | type[BackendEntrypoint]) -> BackendEntrypoint:
    """Select open_dataset method based on current engine."""
    if isinstance(engine, str):
        engines = list_engines()
        if engine not in engines:
            raise ValueError(
                f"unrecognized engine {engine} must be one of: {list(engines)}"
            )
        backend = engines[engine]
    elif isinstance(engine, type) and issubclass(engine, BackendEntrypoint):
        backend = engine()
    else:
        raise TypeError(
            "engine must be a string or a subclass of "
            f"xarray.backends.BackendEntrypoint: {engine}"
        )

    return backend
