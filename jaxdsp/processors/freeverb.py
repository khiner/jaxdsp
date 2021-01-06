import jax.numpy as jnp
from jax import jit, lax

from jaxdsp.processors import lowpass_feedback_comb_filter as comb
from jaxdsp.processors import allpass_filter as allpass

NAME = 'Freeverb'

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

def create_comb_carry(buffer_size, feedback=0.5, damp=0.0):
    return {
        'state': comb.init_state(buffer_size=buffer_size),
        'params': comb.init_params(feedback=feedback, damp=damp),
    }

def create_allpass_carry(buffer_size, feedback=0.5):
    return {
        'state': allpass.init_state(buffer_size=buffer_size),
        'params': allpass.init_params(feedback=feedback),
    }

def init_state():
    return {
        'combs_l': [create_comb_carry(buffer_size) for buffer_size in comb_tunings_l],
        'combs_r': [create_comb_carry(buffer_size + stereo_spread) for buffer_size in comb_tunings_l],
        'allpasses_l': [create_allpass_carry(buffer_size) for buffer_size in allpass_tunings_l],
        'allpasses_r': [create_allpass_carry(buffer_size + stereo_spread) for buffer_size in allpass_tunings_l],
    }

def init_params():
    return {
        'wet': 0.0,
        'dry': 1.0,
        'width': 0.0,
        'damp': 0.0,
        'room_size': 0.0,
    }

def default_target_params():
    return {
        'wet': 0.3,
        'dry': 0.0,
        'width': 1.0,
        'damp': 0.5,
        'room_size': 0.5,
    }

@jit
def tick(carry, x):
    x_r, x_l = jnp.broadcast_to(x, (2,)) # handle stereo or mono in
    x_combined = (x_l + x_r) * fixed_gain

    state = carry['state']
    params = carry['params']

    out_l = 0.0
    out_r = 0.0

    # comb filters in parallel
    for i in range(len(comb_tunings_l)):
        state['combs_l'][i], out_comb_l = comb.tick(state['combs_l'][i], x_combined)
        state['combs_r'][i], out_comb_r = comb.tick(state['combs_r'][i], x_combined)
        out_l += out_comb_l
        out_r += out_comb_r
    # allpasses in series
    for i in range(len(allpass_tunings_l)):
        state['allpasses_l'][i], out_l = allpass.tick(state['allpasses_l'][i], out_l)
        state['allpasses_r'][i], out_r = allpass.tick(state['allpasses_r'][i], out_r)

    wet = params['wet'] * scale_wet
    dry = params['dry'] * scale_dry
    wet_1 = wet * (params['width'] / 2 + 0.5)
    wet_2 = wet * ((1 - params['width']) / 2)

    output_l = out_l * wet_1 + out_r * wet_2 + x_l * dry
    output_r = out_r * wet_1 + out_l * wet_2 + x_r * dry
    return carry, jnp.array([output_l, output_r])

@jit
def tick_buffer(carry, X):
    state = carry['state']
    params = carry['params']

    room_size = (params['room_size'] * scale_room) + offset_room
    damp = params['damp'] * scale_damp
    for comb_l in state['combs_l']:
        comb_l['params']['feedback'] = room_size
        comb_l['params']['damp'] = damp
    for comb_r in state['combs_r']:
        comb_r['params']['feedback'] = room_size
        comb_r['params']['damp'] = damp

    return lax.scan(tick, carry, X)
