import json

from jaxdsp import tracer
from jaxdsp.tracer import trace


# Override trace key
@trace(label="subtract")
def minus(x, y):
    return x - y


result = minus(8, 1) - minus(3, 2) - minus(3, 1)
assert result == 4  # trace shouldn't alter return value

events = tracer.find_events("subtract")
assert len(events) == 3
for event in events:
    print(event.serialize())
    assert type(event.end_time_ms) == int
    assert len(str(event.end_time_ms)) == 13
    assert type(event.duration_ms) == float
    assert event.duration_ms < 1  # sanity check

assert type(tracer.get_events()) == list
assert len(tracer.get_events()) == 3
assert len(tracer.get_events_serialized()) == 3
assert type(json.dumps(tracer.get_events_serialized())) == str  # no errors
assert len(tracer.find_events(label="subtract")) == 3
assert len(tracer.find_events(function_name="minus")) == 3

tracer.clear_events()
assert tracer.get_events() == []
