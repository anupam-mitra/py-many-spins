import time
import uuid


def history_record(ix_time, time_value, extra_fields=None, omit_none=False):
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


def history_records(tlist, extra_fields=None, omit_none=False):
    return [
        history_record(ix_time, time_value, extra_fields, omit_none=omit_none)
        for ix_time, time_value in enumerate(tlist)
    ]
