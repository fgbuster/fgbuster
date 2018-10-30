from fgbuster.pysm_helpers import get_instrument, get_sky
import healpy as hp
import numpy as np
from fgbuster.xForecast import xForecast
import pylab as pl 
nside = 16
# define sky and foregrounds simulations
sky = get_sky(nside, 'd0s0')
# define instrument
instrument = get_instrument(nside, 'litebird')
# get noiseless frequency maps
freq_maps = instrument.observe(sky, write_outputs=False)[0]

# for i in range(5):
# instrument.Noise_Seed = 12345
# np.random.seed(12345)
# freq_maps = instrument.observe(sky, write_outputs=False)[1]
# print freq_maps.shape
# for f in range(freq_maps.shape[0]):
	# hp.mollview(freq_maps[f][1], title=str(f))
# pl.show()
# take only the Q and U maps
freq_maps = freq_maps[:,1:]

# define components used in the modeling
from fgbuster.component_model import CMB, Dust, Synchrotron
components = [CMB( ), Dust( 350.0), Synchrotron( 20.0 )]
# call for xForecast 
res = xForecast(components, instrument, freq_maps, 2, 2*nside-1, 1.0, make_figure=False)
