# Example from doc
# Simple component separation

import healpy as hp
import pysm
from pysm.nominal import models
from pysm.common import loadtxt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator
from fgbuster.observation_helpers import get_instrument, get_sky
from fgbuster.component_model import SemiBlind, CMB, Dust, Synchrotron
#from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.separation_recipies import noise_real_max, test_fisher, _get_alms
from fgbuster.visualization import corner_norm
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.algebra import _mm, _mtmm

nside = 32
lmax = 3*nside-1
npts = 200
cmb_seed = np.arange(npts)
outfile = "scripts/examples/data/noise_CMB_so_nomask.txt"

mask = hp.read_map("/home/clement/Documents/Projets/Physique/Postdoc_APC/Software/fgbuster/fgbuster/templates/HFI_Mask_GalPlane-apo2_2048_R2.00.fits", field=(2))
mask = hp.ud_grade(mask, nside_out=nside)
fsky = float(mask.sum()) / mask.size
instrument = pysm.Instrument(get_instrument('so_la', nside))

cl = np.zeros((2, lmax+1))

# Simulate frequency maps
for seed in cmb_seed:
    cmb = {
        'model': 'taylens',
        'cmb_specs': loadtxt('/home/clement/Documents/Projets/Physique/Postdoc_APC/Software/CAMB_Nov13/fgbuster/test_lenspotentialCls.dat', mpi_comm=None, unpack=True),
        'delens': False,
        'delensing_ells': loadtxt('/home/clement/Documents/Projets/Physique/Postdoc_APC/Software/PySM_public/pysm/template/delens_ells.txt', mpi_comm=None),
        'nside': nside,
        'cmb_seed': seed
    }
    sky_config = {'cmb' : [cmb]} # a la PySM
    sky = pysm.Sky(sky_config)
    freq_maps = instrument.observe(sky, write_outputs=False)[0]
    freq_maps *= mask #apply mask to sky maps
    
    print('Computing alms')
    try:
        assert np.any(instrument.Beams)
    except (KeyError, AssertionError):
        beams = None
    else:  # Deconvolve the beam
        beams = instrument.Beams
    alms = _get_alms(freq_maps, beams, lmax)
    cl += np.sum([hp.sphtfunc.alm2cl(alms[i, ...], lmax=lmax) for i in np.arange(alms.shape[0])], axis=0)[1:2, :]/6.

cl = cl/npts
print(cl)

#np.savetxt(outfile, x)
