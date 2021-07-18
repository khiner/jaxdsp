from jaxdsp.processors import default_param_values, processor_by_name

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
