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
n_comp = 2
lmax = 3 * nside - 1
sky = pysm.Sky(get_sky(nside, 'c1d0'))
instrument = pysm.Instrument(get_instrument('semiblind_test', nside))
freq_maps = instrument.observe(sky, write_outputs=False)[0]
#freq_maps = freq_maps[:, 1:]  # Select polarization

'''
# Get prior cl
print('Computing prior')
templates = '/mnt/PersoPro/Documents/Projets/Physique/Postdoc_APC/Software/fgbuster/fgbuster/templates/Cls_Planck2018_lensed_scalar.fits'
cl_in = hp.read_cl(templates)[1:3,:lmax+1] #Take only polarization
with np.errstate(divide='ignore'):
    EE_in = np.array([np.diag(np.append(1/cl, np.zeros(n_comp-1))) for cl in cl_in[0,:]]) #Should modify here in case several non-blind components
    BB_in = np.array([np.diag(np.append(1/cl, np.zeros(n_comp-1))) for cl in cl_in[1,:]])
    invS_ell = np.stack((EE_in, BB_in), axis=1) #Probably a better way to do that
invS_ell[~np.isfinite(invS_ell)] = 0.
'''

#Inverse noise matrix
bl = [hp.gauss_beam(np.radians(b/60.), lmax=lmax) for b in instrument.Beams]
invN_ell = (np.array(bl) / np.radians(instrument.Sens_P/60.)[:, np.newaxis])**2
invN_ell = np.array([np.diag(invN_ell[:,l]) for l in np.arange(invN_ell.shape[1])])

# Produce dust cl
dust = sky.dust(instrument.Frequencies)
dust_alms=[]
for f, fdata in enumerate(dust):
    dust_alms.append(hp.map2alm(fdata, lmax=lmax)[2,:])
dust_cl = np.array([hp.alm2cl(alm) for alm in dust_alms])
print('dust_cl : ', dust_cl.shape)

#invS_ell, invN_ell = harmonic_semiblind(components, instrument, templates, freq_maps, nside, inv_Nl)
ell = np.arange(2, lmax+1)

#C_ell = np.genfromtxt("../PySM_public/pysm/template/camb_lenspotentialCls.dat", skip_header=1, usecols=(2, 3))
C_ell = np.genfromtxt("../CAMB_Nov13/fgbuster/test_lenspotentialCls.dat", usecols=(2, 3))
print('C_ell : ', C_ell.shape)

#Plot things
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.weight"] = "light"
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(12 ,10))
#gs1 = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
#gs1.update(left=0.125, right=0.95, wspace=0.05)
#ax1 = plt.subplot(gs1[0])
#ax2 = plt.subplot(gs1[1])

plt.yscale('log')
plt.xscale('log')
#plt.plot(ell, ell*(ell+1)/(2*np.pi*invS_ell[2:,1,0,0]), 'g', label='Prior', linestyle='--')
plt.plot(ell, ell*(ell+1)/(2*np.pi*invN_ell[2:,0,0]), 'r', label=r'Noise at $\nu = 100 \ \mathrm{GHz}$', linestyle='--')
plt.plot(ell, ell*(ell+1)*dust_cl[0,2:]/(2*np.pi), 'r', label=r'Dust at $\nu = 100 \ \mathrm{GHz}$')
plt.plot(ell, ell*(ell+1)/(2*np.pi*invN_ell[2:,1,1]), 'b', label=r'Noise at $\nu = 145 \ \mathrm{GHz}$', linestyle='--')
plt.plot(ell, ell*(ell+1)*dust_cl[1,2:]/(2*np.pi), 'b', label=r'Dust at $\nu = 145 \ \mathrm{GHz}$')
plt.plot(ell, ell*(ell+1)/(2*np.pi*invN_ell[2:,2,2]), 'k', label=r'Noise at $\nu = 200 \ \mathrm{GHz}$', linestyle='--')
plt.plot(ell, ell*(ell+1)*dust_cl[2,2:]/(2*np.pi), 'k', label=r'Dust at $\nu = 200 \ \mathrm{GHz}$')
plt.plot(ell, C_ell[:lmax-1,1], 'orange', label=r'CMB with $r= 10^{-3}$')
#plt.plot(ell, C_ell_test[:lmax-1,0], 'purple', label='CMB test')
plt.legend(loc="lower right", prop={'size':15})
plt.ylabel(r'$\frac{\ell \left( \ell + 1 \right)}{2\pi}C_{\ell}^{BB}$', fontsize=25, labelpad=15)
plt.xlabel(r'$\ell$', fontsize=25, labelpad=15)
plt.tick_params(axis='both', which='major', labelsize=12, length=7, width=1.2)
plt.ylim([5e-8, 20])
#plt.xlim([-6.5, 14])
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

#fig.tight_layout()
fig.savefig('scripts/examples/plots/comparison_cl_trueS.png')
#plt.show()
