# Example from doc
# Simple component separation

import healpy as hp
import pysm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from fgbuster.observation_helpers import get_instrument, get_sky
from fgbuster.component_model import SemiBlind, CMB, Dust, Synchrotron
#from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.separation_recipies import basic_comp_sep, _get_alms, harmonic_semiblind, semiblind
from fgbuster.visualization import corner_norm
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.algebra import _mm, _mtmm

# Simulate your frequency maps
nside = 32
lmax = 3*nside-1

# Define sky configuration
#sky = pysm.Sky(get_sky(nside, 'c1d0')) # a la fgbuster
cmb_seed = 1111
cmb = models("c1", nside, cmb_seed)
dust = models("d0", nside)
sky_config = {'cmb' : cmb, 'dust' : dust} # a la PySM
sky = pysm.Sky(sky_config)
instrument = pysm.Instrument(get_instrument('semiblind_test', nside))
freq_maps = instrument.observe(sky, write_outputs=False)[0]
#freq_maps = freq_maps[:, 1:]  # Select polarization
#print(sky.Components)

# Define what you fit for
components = [CMB(), Dust(150.)]#, Synchrotron(20.)]
#components = [CMB(), Dust(150., 1.54, 20.), Synchrotron(20., 70.)]

#Inverse noise matrix
bl = [hp.gauss_beam(np.radians(b/60.), lmax=3*nside-1) for b in instrument.Beams]
inv_Nl = (np.array(bl) / np.radians(instrument.Sens_P/60.)[:, np.newaxis])**2
inv_Nl = np.array([np.diag(inv_Nl[:,l]) for l in np.arange(inv_Nl.shape[1])])#[2:,np.newaxis,:,:]
inv_N = np.diag(hp.nside2resol(nside, arcmin=True) / (instrument.Sens_P))**2

#CMB prior covariance matrix
#templates = '/mnt/PersoPro/Documents/Projets/Physique/Postdoc_APC/Software/fgbuster/fgbuster/templates/Cls_Planck2018_lensed_scalar.fits'
templates = np.swapaxes(np.genfromtxt("../CAMB_Nov13/fgbuster/test_lenspotentialCls.dat", usecols=(1, 2, 3)), 0, 1)

# Component separation
#result = basic_comp_sep(components, instrument, freq_maps)

x, y, result = harmonic_semiblind(components, instrument, templates, freq_maps, nside, inv_Nl)
#x, y, result = semiblind(components, instrument, templates, freq_maps, nside, inv_N)


outfile = "scripts/examples/data/trueS_full_noise_zoom.txt"
#outfile = "scripts/examples/data/noS_full_pix.txt"
#outfile = "scripts/examples/data/S_100_145_200_noise_zoom.txt"
np.savetxt(outfile, (x, y, result))


#Explore the results
#print(result.params)
#print(result.x)

#corner_norm(result.x, result.Sigma, labels=result.params)

#print(result.s.shape)

#hp.mollview(result.s[0,1], title='CMB')
#hp.mollview(result.s[1,1], title='Dust', norm='hist')
#hp.mollview(result.s[2,1], title='Synchrotron', norm='hist')
#plt.show()
