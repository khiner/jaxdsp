import time
from functools import wraps
from collections import deque

from inspect import iscoroutinefunction
from contextlib import contextmanager


class TraceEvent:
    def __init__(
        self,
        function,
        start_time_ms=None,
        end_time_ms=None,
        duration_ms=None,
        label=None,
    ):
        self.function_name = function.__name__
        self.label = label or self.function_name
        self.start_time_ms = start_time_ms
        self.end_time_ms = end_time_ms
        self.duration_ms = duration_ms

    def serialize(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


# All trace events are added to a qlobal queue.
# Client is responsible for any longer-term storage.
trace_events = deque(maxlen=100)


# Milliseconds since epoch
def time_ms():
    return time.time_ns() // 1_000_000


# TODO think about the need for [block_when_ready](https://jax.readthedocs.io/en/latest/async_dispatch.html)
#  for measuring `jit`ed jax fns.


# Technique for optional named decorator parameters is from:
#   https://stackoverflow.com/a/60832711/780425
# Using `@contextmanager` to allow decorating either async or non-async functions is from:
#   https://gist.github.com/anatoly-kussul/f2d7444443399e51e2f83a76f112364d/ff1f94b1bd07741ce209cc61832f920adb49aedf
def trace(f_py=None, label=None):
    """
    Append a `TraceEvent` after completion of every decorated function call.
    If no value is provided to the (keyword-only, non-positional) `label` parameter, the `label` property of the event
    is set to the name of the decorated function.

    See `/tests/jaxdsp/chart_event_accumulators/test_tracer.py` for examples.
    """

    assert callable(f_py) or f_py is None

    def decorator(function):
        @contextmanager
        def wrapper(inner_function):
            start_time_ms = time.perf_counter()
            trace_events.append(
                TraceEvent(
                    inner_function,
                    start_time_ms=time_ms(),
                    label=label,
                )
            )
            yield
            trace_events.append(
                TraceEvent(
                    inner_function,
                    end_time_ms=time_ms(),
                    duration_ms=(time.perf_counter() - start_time_ms) * 1_000,
                    label=label,
                )
            )

        @wraps(function)
        def sync_wrapper(*args, **kwargs):
            with wrapper(function):
                return function(*args, **kwargs)

        @wraps(function)
        async def async_wrapper(*args, **kwargs):
            with wrapper(function):
                return await function(*args, **kwargs)

        return async_wrapper if iscoroutinefunction(function) else sync_wrapper

    return decorator(f_py) if callable(f_py) else decorator


# TODO move these into static methods on a new `TraceEvents` class. (E.g. `TraceEvents.find(label, function_name)`)


def find_events(label=None, function_name=None) -> list[TraceEvent]:
    return [
        event
        for event in trace_events
        if (label and event.label == label)
        or (function_name and event.function_name == function_name)
    ]


def get_events() -> list[TraceEvent]:
    return list(trace_events)


def get_events_serialized() -> list[dict]:
    return [trace_event.serialize() for trace_event in trace_events]


def clear_events():
    trace_events.clear()
