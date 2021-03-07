import jax.numpy as jnp


class Param:
    def __init__(
        self, name, default_value=0.0, min_value=0.0, max_value=1.0, log_scale=False
    ):
        self.name = name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.log_scale = log_scale

    def serialize(self):
        serialized = self.__dict__
        if isinstance(self.default_value, jnp.DeviceArray):
            serialized["default_value"] = self.default_value.tolist()
        return serialized
