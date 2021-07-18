from jaxdsp.processors import (
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


def default_param_values(processor, processor_names=None):
    if not processor and not processor_names:
        return None

    if processor_names:
        return [
            default_param_values(processor_by_name[processor_name])
            for processor_name in processor_names
        ]
    return {param.name: param.default_value for param in processor.PARAMS}


def id_for_processor(processor):
    return all_processors.index(processor)


def serialize_processor(processor, params=None):
    if not processor:
        return None

    return {
        "name": processor.NAME,
        "param_definitions": [param.serialize() for param in processor.PARAMS],
        "presets": processor.PRESETS,
        "params": params or default_param_values(processor),
    }


def params_to_unit_scale(params, processor_name):
    processor = processor_by_name[processor_name]
    return (
        {
            param.name: param.to_unit_scale(params[param.name])
            for param in processor.PARAMS
        }
        if processor
        else {}
    )


def params_from_unit_scale(params, processor_name):
    processor = processor_by_name[processor_name]
    return (
        {
            param.name: param.from_unit_scale(params[param.name])
            for param in processor.PARAMS
        }
        if processor
        else {}
    )
