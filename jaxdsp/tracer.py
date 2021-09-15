import json
import time
from functools import wraps
from collections import deque

# In seconds
execution_durations = {}


# Technique for optional named decorator parameters is from https://stackoverflow.com/a/60832711/780425
def trace(f_py=None, key=None, limit=1_000):
    """
    Time the function and store the execution duration, in seconds, in the global `execution_durations` dictionary.
    The duration will be associated with the (optional) key provided to the decorator via the (keyword-only,
    non-positional) 'key' parameter. If no key is provided, the name of the decorated function name will be used.
    The (optional) `limit` parameter specifies how many of the most recent results are stored for this key (defaults
    to the most recent 1k values per-key).

    Example:
        from jaxdsp import tracer
        from jaxdsp.tracer import trace

        # Track with an overridden key:
        @trace(key='subtract')
        def minus(x, y):
            return x - y

        result = minus(8, 1) - minus(3,2) - minus(3, 1)
        assert result == 4
        print(timing.get('subtract'))
        > [3.2569999994791488e-06, 8.139999998491021e-07, 6.259999985047671e-07]
        print(tracer.get_all())
        > {"subtract": [3.304999999897973e-06, 8.449999988613399e-07, 6.009999999889715e-07]}
        tracer.clear()
        print(timing.get_all())
        > {}

        # Use the function name as the key, and only store the most recent 2 values for this function.
        @trace(limit=2)
        def minus(x, y):
            return x - y

        result = minus(8, 1) - minus(3,2) - minus(3, 1)
        assert result == 4
        print(tracer.get('minus')) # Assuming the execution times were exactly the same for illustration
        > [8.139999998491021e-07, 6.259999985047671e-07]
        print(tracer.get_all())
        > {"minus": [8.139999998491021e-07, 6.259999985047671e-07]}
    """

    assert callable(f_py) or f_py is None

    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            effective_key = key or func.__name__
            if not execution_durations.get(effective_key):
                execution_durations[effective_key] = deque([], limit)

            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            execution_duration = time.perf_counter() - start_time
            epoch_millis = time.time_ns() // 1_000_000
            execution_durations[effective_key].append(
                [epoch_millis, execution_duration]
            )

            return value

        return wrapper

    return _decorator(f_py) if callable(f_py) else _decorator


def get(key):
    return list(execution_durations[key])


def get_all():
    return {key: list(results) for (key, results) in execution_durations.items()}


def clear():
    execution_durations.clear()
