from jaxdsp.processors import (
    allpass_filter,
    clip,
    delay_line,
    fir_filter,
    freeverb,
    iir_filter,
    biquad_lowpass,
    lowpass_feedback_comb_filter,
    sine_wave,
)

all_processors = [
    allpass_filter,
    clip,
    delay_line,
    fir_filter,
    freeverb,
    iir_filter,
    biquad_lowpass,
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
    if not params or not processor_name:
        return None

    if isinstance(processor_name, list):
        return [
            params_to_unit_scale(processor_params, processor_name)
            for processor_params, processor_name in zip(params, processor_name)
        ]

    processor = processor_by_name[processor_name]
    return {
        param.name: param.to_unit_scale(params[param.name])
        for param in processor.PARAMS
    }


def params_from_unit_scale(params, processor_name):
    if not params or not processor_name:
        return None

    if isinstance(processor_name, list):
        return [
            params_from_unit_scale(processor_params, processor_name)
            for processor_params, processor_name in zip(params, processor_name)
        ]

    processor = processor_by_name[processor_name]
    return {
        param.name: param.from_unit_scale(params[param.name])
        for param in processor.PARAMS
    }


def processor_names_from_graph_config(graph_config):
    if not graph_config:
        return None

    return [
        processor_names_from_graph_config(processor_config)
        if isinstance(processor_config, list)
        else processor_config["name"]
        for processor_config in graph_config
    ]


def processor_names_to_graph_config(processor_names):
    if not processor_names:
        return None

    return [
        processor_names_to_graph_config(processor_name)
        if isinstance(processor_name, list)
        else {"name": processor_name}
        for processor_name in processor_names
    ]


def processors_to_graph_config(processors):
    return [
        processors_to_graph_config(processor)
        if isinstance(processor, list)
        else {"name": processor.NAME}
        for processor in processors
    ]


def get_graph_params(graph_config):
    if not graph_config:
        return None

    return [
        get_graph_params(processor_config)
        if isinstance(processor_config, list)
        else (
            processor_config["params"]
            if "params" in processor_config
            else default_param_values(processor_by_name[processor_config["name"]])
        )
        for processor_config in graph_config
    ]


def init_graph_state(graph_config):
    if not graph_config:
        return None

    return [
        init_graph_state(processor_config)
        if isinstance(processor_config, list)
        else processor_by_name[processor_config["name"]].init_state()
        for processor_config in graph_config
    ]


# Returns (params, state)
def graph_config_to_carry(graph_config):
    if not graph_config:
        return None, None

    return get_graph_params(graph_config), init_graph_state(graph_config)


def set_state_recursive(state, key, value):
    if not state:
        return

    if isinstance(state, list):
        for state_item in state:
            set_state_recursive(state_item, key, value)
    else:
        state[key] = value
