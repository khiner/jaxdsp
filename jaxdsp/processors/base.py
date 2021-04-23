# TODO not used, keeping around for now in case I remember why I added but I think it's outdated.
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
    def __init__(self, state_init, title=None):
        self.state_init = state_init
        self.title = title
