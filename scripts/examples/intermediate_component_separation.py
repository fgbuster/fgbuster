import pylab
pylab.rcParams['figure.figsize'] = 12, 16

import healpy as hp
import pysm
import matplotlib.pyplot as plt

from fgbuster.observation_helpers import get_instrument, get_sky  # Predefined instrumental and sky-creation configurations
from fgbuster.visualization import corner_norm

# Imports needed for component separation
from fgbuster import (CMB, Dust, Synchrotron,  # sky-fitting model
                      basic_comp_sep)  # separation routine



NSIDE = 64
sky_conf_simple = get_sky(NSIDE, 'c1d0s0')
instrument = pysm.Instrument(get_instrument('cmbs4', NSIDE))
freq_maps_simple, noise = instrument.observe(pysm.Sky(sky_conf_simple), write_outputs=False)


freq_maps_simple = freq_maps_simple[:, 1:]  # Select polarization


NSIDE_PATCH = 8
sky_conf_vary = get_sky(NSIDE, 'c1d1s1')


for comp, param in [('dust', 'spectral_index'),
                    ('dust', 'temp'),
                    ('synchrotron', 'spectral_index')
                   ]:
    spectral_param = sky_conf_vary[comp][0][param]
    spectral_param[:] = hp.ud_grade(hp.ud_grade(spectral_param, NSIDE_PATCH),
                                    NSIDE)


comp = 'dust'
param = 'spectral_index'
hp.mollview(sky_conf_simple[comp][0][param], sub=(1,3,1), title='Constant index')
hp.mollview(sky_conf_vary[comp][0][param], sub=(1,3,2), title='Varying indices')
hp.mollview(get_sky(NSIDE, 'c1d1s1')[comp][0][param], sub=(1,3,3), title='Full resolution indices')


freq_maps_vary, _ = instrument.observe(pysm.Sky(sky_conf_vary), write_outputs=False)
freq_maps_vary = freq_maps_vary[:, 1:] # Select polarization


components = [CMB(), Dust(353.), Synchrotron(23.)]


# The starting point of the fit is the pysm default value, so let's shift it
components[1].defaults = [1.6, 22.]
components[2].defaults = [-2.7]


result = basic_comp_sep(components, instrument, freq_maps_simple,
                        options=dict(disp=True),  # verbose output
                        )


import numpy as np
inputs = [sky_conf_simple[comp][0][param][0]
          for comp, param in [('dust', 'spectral_index'),
                              ('dust', 'temp'),
                              ('synchrotron', 'spectral_index')]
         ]
print("%-20s\t%s\t%s" % ('', 'Estimated', 'Input'))
for param, val, ref in zip(result.params, result.x, inputs):
    print("%-20s\t%f\t%f" % (param, val, ref))


hp.mollview(result.s[0,1], title='CMB', sub=(2,3,1))
hp.mollview(result.s[1,1], title='Dust', norm='hist', sub=(2,3,2))
hp.mollview(result.s[2,1], title='Synchrotron', norm='hist', sub=(2,3,3))


hp.mollview(result.s[1,1]
            - sky_conf_simple['dust'][0]['A_U'] * pysm.convert_units('K_RJ', 'K_CMB', 353.),
            title='Dust', norm='hist', sub=(3,2,1))
hp.mollview(result.s[2,1]
            - sky_conf_simple['synchrotron'][0]['A_U'] * pysm.convert_units('K_RJ', 'K_CMB', 23.),
            title='Synchrotron', norm='hist', sub=(3,2,2))


corner_norm(result.x, result.Sigma, labels=result.params, truths=inputs)


plt.show()
