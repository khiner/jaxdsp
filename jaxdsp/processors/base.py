# TODO not used, keeping around for now in case I remember why I added but I think it's outdated.
# https://stackoverflow.com/a/20666342/780425
from jaxdsp.processors import default_param_values, processor_by_name, serial_processors


def deep_merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value

    return destination

# Returns (processor_definition, processor_params, processor_state)
def processor_triplet_for_config(processor_config):
    if not processor_config:
        return None, None, None

    if isinstance(processor_config, list):
        if len(processor_config) == 0:
            return None, None, None

        params = [processor["params"] if "params" in processor else default_param_values(processor_by_name[processor["name"]]) for processor in processor_config]
        state = [processor_by_name[processor["name"]].init_state() for processor in processor_config]
        return serial_processors, params, state

    processor = processor_by_name.get(processor_config["name"])
    return processor, processor_config["params"], processor.init_state()
