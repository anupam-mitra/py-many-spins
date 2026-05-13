import time
import uuid
from typing import Any

def history_record(
    ix_time: int, time_value: float, extra_fields: dict[str, Any] | None = None, omit_none: bool = False
) -> dict[str, Any]:
    """
    Create a record for a single time step.

    Parameters
    ----------
    ix_time : int
        The index of the time step.
    time_value : float
        The time value of the step.
    extra_fields : dict[str, Any], optional
        Additional fields to include in the record, by default None.
    omit_none : bool, optional
        Whether to omit fields with None values, by default False.

    Returns
    -------
    dict[str, Any]
        The created record.
    """
    record = {
        "ix_time": ix_time,
        "time": time_value,
        "uuid_str": "%s" % uuid.uuid4(),
        "walltime": time.time(),
    }
    for key, value in dict(extra_fields or {}).items():
        if omit_none and value is None:
            continue
        record[key] = value
    return record


def history_records(
    tlist: list[float] | Any, extra_fields: dict[str, Any] | None = None, omit_none: bool = False
) -> list[dict[str, Any]]:
    """
    Create a list of records for a given time list.

    Parameters
    ----------
    tlist : list[float] | Any
        The time grid (iterable of time values).
    extra_fields : dict[str, Any], optional
        Additional fields to include in each record, by default None.
    omit_none : bool, optional
        Whether to omit fields with None values, by default False.

    Returns
    -------
    list[dict[str, Any]]
        The list of generated records.
    """
    return [
        history_record(ix_time, time_value, extra_fields, omit_none=omit_none)
        for ix_time, time_value in enumerate(tlist)
    ]

