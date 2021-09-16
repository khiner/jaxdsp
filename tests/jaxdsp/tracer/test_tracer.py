from jaxdsp import tracer
from jaxdsp.tracer import trace


# Override trace key
@trace(key="subtract")
def minus(x, y):
    return x - y


result = minus(8, 1) - minus(3, 2) - minus(3, 1)
assert result == 4  # trace shouldn't alter return value

traces = tracer.get("subtract")
assert len(traces) == 3
for trace in traces:
    assert len(trace) == 2
    finish_time_ms, execution_duration_ms = trace
    print(finish_time_ms, execution_duration_ms)
    assert type(finish_time_ms) == int
    assert type(execution_duration_ms) == float
    assert len(str(finish_time_ms)) == 13
    assert execution_duration_ms < 1  # sanity check

assert type(tracer.get_all()) == dict
assert len(tracer.get_all().keys()) == 1

tracer.clear()
assert tracer.get_all() == {}
