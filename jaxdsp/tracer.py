import time
from functools import partial, wraps
from collections import defaultdict, deque

from inspect import iscoroutinefunction
from contextlib import contextmanager

# This tracer is intended to publish its in-memory data on stream where a client can consume it,
# so values are only stored in memory here to allow:
# * Publishing to the stream in buffers
# * Resiliency to back-pressure from downstream
# Client is responsible for longer-term storage as needed.
MAX_IN_MEMORY_SERIES_LENGTH = 100

trace_series_for_key = defaultdict(partial(deque, maxlen=MAX_IN_MEMORY_SERIES_LENGTH))


# Milliseconds since epoch
def time_ms():
    return time.time_ns() // 1_000_000


# TODO think about the need for [block_when_ready](https://jax.readthedocs.io/en/latest/async_dispatch.html)
#  for measuring `jit`ed jax fns.

# Technique for optional named decorator parameters is from:
#   https://stackoverflow.com/a/60832711/780425
# Using `@contextmanager` to allow decorating either async or non-async functions is from:
#   https://gist.github.com/anatoly-kussul/f2d7444443399e51e2f83a76f112364d/ff1f94b1bd07741ce209cc61832f920adb49aedf
def trace(f_py=None, key=None):
    """
    Annotation for function timing. A `[finish_time_ms, execution_duration_ms]` pair will be associated with the
    optional (keyword-only, non-positional) `key` parameter.
    If no key is provided, it will use the name of the decorated function.

    See `/tests/jaxdsp/tracer/test_tracer.py` for examples.
    """

    assert callable(f_py) or f_py is None

    def decorator(func):
        @contextmanager
        def wrapper(inner_func):
            start_time = time.perf_counter()
            yield
            execution_duration_ms = (time.perf_counter() - start_time) * 1_000
            finish_time_ms = time_ms()
            trace_series_for_key[key or inner_func.__name__].append(
                [finish_time_ms, execution_duration_ms]
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
