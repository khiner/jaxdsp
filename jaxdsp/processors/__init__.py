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


def default_param_values(processor):
    return (
        {param.name: param.default_value for param in processor.PARAMS}
        if processor
        else None
    )


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

# Returns (params, state)
def processor_config_to_carry(processor_config):
    if not processor_config:
        return None, None
    params = [
        processor["params"]
        if "params" in processor
        else default_param_values(processor_by_name[processor["name"]])
        for processor in processor_config
    ]
    state = [
        processor_by_name[processor["name"]].init_state()
        for processor in processor_config
    ]
    return params, state
