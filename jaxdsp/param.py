import jax.numpy as jnp


class Param:
    def __init__(self, name, default_value=0.0, min_value=0.0, max_value=1.0):
        self.name = name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value

    def serialize(self):
        return {
            "name": self.name,
            "default_value": self.default_value.tolist()
            if isinstance(self.default_value, jnp.DeviceArray)
            else self.default_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }
