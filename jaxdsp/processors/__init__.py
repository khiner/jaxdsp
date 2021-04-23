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


def param_by_name(processor_name):
    processor = processor_by_name.get(processor_name)
    return {param.name: param for param in processor.PARAMS} if processor else {}


def params_to_unit_scale(params, processor_name):
    return {
        name: params_to_unit_scale(value, name)
        if name in processor_by_name
        else param_by_name(processor_name)[name].to_unit_scale(value)
        for name, value in params.items()
    }


def params_from_unit_scale(params, processor_name):
    return {
        name: params_from_unit_scale(value, name)
        if name in processor_by_name
        else param_by_name(processor_name)[name].from_unit_scale(value)
        for name, value in params.items()
    }
