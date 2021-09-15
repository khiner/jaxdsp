import time
from functools import wraps
from collections import deque
from inspect import iscoroutinefunction
from contextlib import contextmanager

trace_series_for_key = {}


# Technique for optional named decorator parameters is from:
#   https://stackoverflow.com/a/60832711/780425
# Using `@contextmanager` to allow decorating either async or non-async functions is from:
#   https://gist.github.com/anatoly-kussul/f2d7444443399e51e2f83a76f112364d/ff1f94b1bd07741ce209cc61832f920adb49aedf
def trace(f_py=None, key=None, limit=1_000):
    """
    Annotation for function timing. A `[finish_time_ms, execution_duration_ms]` pair will be associated with the
    optional (keyword-only, non-positional) `key` parameter.
    If no key is provided, it will use the name of the decorated function.
    The optional `limit` parameter (default=1k) specifies how many of the most recent results should be stored
    for this key.

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
        > [
            [1631683258589, 3.2569999994791488e-03],
            [1631683259580, 8.139999998491021e-04],
            [1631683261680, 6.259999985047671e-04]
          ]

        print(tracer.get_all())
        > {
            "subtract": [
              [1631683258589, 3.2569999994791488e-03],
              [1631683259580, 8.139999998491021e-04],
              [1631683261680, 6.259999985047671e-04]
            ]
          }

        tracer.clear()
        print(timing.get_all())
        > {}

        # Use the function name as the key, and only store the most recent 2 values for this function.
        @trace(limit=2)
        def minus(x, y):
            return x - y

        result = minus(8, 1) - minus(3,2) - minus(3, 1)
        assert result == 4

        print(tracer.get('minus')) # Assuming the values were exactly the same for illustration
        > [
            [1631683259580, 8.139999998491021e-04],
            [1631683261680, 6.259999985047671e-04]
          ]
    """

    assert callable(f_py) or f_py is None

    def decorator(func):
        @contextmanager
        def wrapper(inner_func):
            start_time = time.perf_counter()
            yield
            execution_duration_s = time.perf_counter() - start_time
            epoch_millis = time.time_ns() // 1_000_000
            effective_key = key or inner_func.__name__
            if not trace_series_for_key.get(effective_key):
                trace_series_for_key[effective_key] = deque([], limit)
            trace_series_for_key[effective_key].append(
                [epoch_millis, execution_duration_s * 1_000]
            )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with wrapper(func):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with wrapper(func):
                return await func(*args, **kwargs)

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    return decorator(f_py) if callable(f_py) else decorator


def get(key):
    return list(trace_series_for_key[key])


def get_all():
    return {key: list(results) for (key, results) in trace_series_for_key.items()}


def clear():
    trace_series_for_key.clear()
