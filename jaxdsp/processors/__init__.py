from jaxdsp.processors import (
    base,
    allpass_filter,
    clip,
    delay_line,
    feedforward_delay,
    fir_filter,
    freeverb,
    iir_filter,
    lowpass_feedback_comb_filter,
    sine_wave,
)

all_processors = [
    allpass_filter,
    clip,
    delay_line,
    feedforward_delay,
    fir_filter,
    freeverb,
    iir_filter,
    lowpass_feedback_comb_filter,
    sine_wave,
]
processor_by_name = {processor.NAME: processor for processor in all_processors}
empty_carry = {"state": None, "params": None}


def serialize_processor(processor, params=None):
    if not processor:
        return None

    return {
        "name": processor.NAME,
        "param_definitions": [param.serialize() for param in processor.PARAMS],
        "presets": processor.PRESETS,
        "params": params or base.default_param_values(processor.PARAMS),
    }
