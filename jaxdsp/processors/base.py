def default_param_values(params):
    return {param.name: param.default_value for param in params}

# https://stackoverflow.com/a/20666342/780425
def deep_merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value

    return destination

class Config:
    def __init__(self, state_init, params_init, params_target, title=None):
        self.state_init = state_init
        self.params_init = params_init
        self.params_target = params_target
        self.title = title
