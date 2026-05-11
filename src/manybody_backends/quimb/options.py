def _selected_options(params, keys, defaults=None):
    params = params or {}
    options = {}
    for key in keys:
        if params.get(key) is not None:
            options[key] = params[key]
    for key, value in dict(defaults or {}).items():
        options.setdefault(key, value)
    return options


def tebd_init_options(tebd_params):
    """Return options for ``quimb.tensor.TEBD`` construction."""
    return _selected_options(
        tebd_params,
        ("dt", "tol", "progbar"),
        defaults={"progbar": False},
    )


def tebd_evolution_options(tebd_params):
    """Return options for ``TEBD.at_times`` evolution."""
    return _selected_options(
        tebd_params,
        ("dt", "tol", "order", "progbar"),
        defaults={"progbar": False},
    )


def split_options(trunc_params):
    """Translate workflow truncation settings to Quimb split options."""
    params = trunc_params or {}
    options = {}
    if params.get("chi_max") is not None:
        options["max_bond"] = params["chi_max"]
    if params.get("cutoff") is not None:
        options["cutoff"] = params["cutoff"]
    elif params.get("svd_min") is not None:
        options["cutoff"] = params["svd_min"]
    return options


def gate_options(split_opts):
    """Return Quimb gate options supported by local one-site gates."""
    return _selected_options(split_opts, ("max_bond", "cutoff"))
