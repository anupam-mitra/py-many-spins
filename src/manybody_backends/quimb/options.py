from typing import Any

def _selected_options(params: dict[str, Any] | None, keys: tuple[str, ...], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Filter parameters to a set of keys, applying defaults if values are missing.

    Parameters
    ----------
    params : dict[str, Any] | None
        The input parameters.
    keys : tuple[str, ...]
        The keys to extract.
    defaults : dict[str, Any] | None, optional
        Default values for the keys, by default None.

    Returns
    -------
    dict[str, Any]
        The filtered options.
    """
    params = params or {}
    options = {}
    for key in keys:
        if params.get(key) is not None:
            options[key] = params[key]
    for key, value in dict(defaults or {}).items():
        options.setdefault(key, value)
    return options


def tebd_init_options(tebd_params: dict[str, Any] | None) -> dict[str, Any]:
    """
    Return options for ``quimb.tensor.TEBD`` construction.

    Parameters
    ----------
    tebd_params : dict[str, Any] | None
        The TEBD parameters.

    Returns
    -------
    dict[str, Any]
        The initialization options.
    """
    return _selected_options(
        tebd_params,
        ("dt", "tol", "progbar"),
        defaults={"progbar": False},
    )


def tebd_evolution_options(tebd_params: dict[str, Any] | None) -> dict[str, Any]:
    """
    Return options for ``TEBD.at_times`` evolution.

    Parameters
    ----------
    tebd_params : dict[str, Any] | None
        The TEBD parameters.

    Returns
    -------
    dict[str, Any]
        The evolution options.
    """
    return _selected_options(
        tebd_params,
        ("dt", "tol", "order", "progbar"),
        defaults={"progbar": False},
    )


def split_options(trunc_params: dict[str, Any] | None) -> dict[str, Any]:
    """
    Translate workflow truncation settings to Quimb split options.

    Parameters
    ----------
    trunc_params : dict[str, Any] | None
        The truncation parameters.

    Returns
    -------
    dict[str, Any]
        The Quimb split options.
    """
    params = trunc_params or {}
    options = {}
    if params.get("chi_max") is not None:
        options["max_bond"] = params["chi_max"]
    if params.get("cutoff") is not None:
        options["cutoff"] = params["cutoff"]
    elif params.get("svd_min") is not None:
        options["cutoff"] = params["svd_min"]
    return options


def gate_options(split_opts: dict[str, Any]) -> dict[str, Any]:
    """
    Return Quimb gate options supported by local one-site gates.

    Parameters
    ----------
    split_opts : dict[str, Any]
        The split options.

    Returns
    -------
    dict[str, Any]
        The gate options.
    """
    return _selected_options(split_opts, ("max_bond", "cutoff"))

