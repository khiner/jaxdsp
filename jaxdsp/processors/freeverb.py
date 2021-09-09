import jax.numpy as jnp
from jax import jit, lax

from jaxdsp.param import Param
from jaxdsp.processors import allpass_filter as allpass
from jaxdsp.processors import lowpass_feedback_comb_filter as comb

NAME = "Freeverb"
PARAMS = [
    Param("wet", 0.0),
    Param("dry", 1.0),
    Param("width", 0.0),
    Param("damp", 0.0),
    Param("room_size", 0.0, 0.0, 1.2),
]
PRESETS = {
    "flat_space": {
        "wet": 0.3,
        "dry": 0.6,
        "width": 0.5,
        "damp": 0.3,
        "room_size": 1.055,
    },
    "expanding_space": {
        "wet": 0.33,
        "dry": 0.0,
        "width": 0.5,
        "damp": 0.1,
        "room_size": 1.078,
    },
}

fixed_gain = 0.015
scale_wet = 3.0
scale_dry = 2.0
scale_damp = 0.4
scale_room = 0.28
offset_room = 0.7

comb_tunings_l = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
allpass_tunings_l = [556, 441, 341, 225]
stereo_spread = 23


# comb_tunings_l = [1, 2, 3, 4, 5, 6, 7, 9]
# allpass_tunings_l = [2, 3, 4, 5]


def init_comb_carry(buffer_size):
    return {"feedback": 0.5, "damp": 0.0}, comb.init_state(buffer_size=buffer_size)


def init_allpass_carry(buffer_size):
    return {"feedback": 0.5}, allpass.init_state(buffer_size=buffer_size)


def init_state():
    return {
        "combs_l": [init_comb_carry(buffer_size) for buffer_size in comb_tunings_l],
        "combs_r": [
            init_comb_carry(buffer_size + stereo_spread)
            for buffer_size in comb_tunings_l
        ],
        "allpasses_l": [
            init_allpass_carry(buffer_size) for buffer_size in allpass_tunings_l
        ],
        "allpasses_r": [
            init_allpass_carry(buffer_size + stereo_spread)
            for buffer_size in allpass_tunings_l
        ],
    }


@jit
def tick(carry, x):
    params, state = carry

    x_r, x_l = jnp.broadcast_to(x, (2,))  # handle stereo or mono in
    x_combined = (x_l + x_r) * fixed_gain

    out_l = 0.0
    out_r = 0.0

    # comb filters in parallel
    for i in range(len(comb_tunings_l)):
        state["combs_l"][i], out_comb_l = comb.tick(state["combs_l"][i], x_combined)
        state["combs_r"][i], out_comb_r = comb.tick(state["combs_r"][i], x_combined)
        out_l += out_comb_l
        out_r += out_comb_r
    # allpasses in series
    for i in range(len(allpass_tunings_l)):
        state["allpasses_l"][i], out_l = allpass.tick(state["allpasses_l"][i], out_l)
        state["allpasses_r"][i], out_r = allpass.tick(state["allpasses_r"][i], out_r)

    wet = params["wet"] * scale_wet
    dry = params["dry"] * scale_dry
    wet_1 = wet * (params["width"] / 2 + 0.5)
    wet_2 = wet * ((1 - params["width"]) / 2)

    output_l = out_l * wet_1 + out_r * wet_2 + x_l * dry
    output_r = out_r * wet_1 + out_l * wet_2 + x_r * dry
    return carry, jnp.array([output_l, output_r])


@jit
def tick_buffer(carry, X):
    params, state = carry

    room_size = (params["room_size"] * scale_room) + offset_room
    damp = params["damp"] * scale_damp
    for comb_l in state["combs_l"]:
        comb_l[0]["feedback"] = room_size
        comb_l[0]["damp"] = damp
    for comb_r in state["combs_r"]:
        comb_r[0]["feedback"] = room_size
        comb_r[0]["damp"] = damp

    carry_out, Y = lax.scan(tick, carry, X)
    return carry_out, Y.T
