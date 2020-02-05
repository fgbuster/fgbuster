# Example from doc
# Simple component separation

from __future__ import print_function
import healpy as hp
import pysm
from scipy import interpolate
from scipy.stats import norm
from pysm.nominal import models
from pysm.common import loadtxt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator
import time
from fgbuster.observation_helpers import get_instrument, get_sky
from fgbuster.component_model import SemiBlind, CMB, Dust, Synchrotron
#from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.separation_recipies import noise_real_max, test_fisher
from fgbuster.visualization import corner_norm
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.algebra import _mm, _mtmm
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
import IPython
import numpy as np
from corner import corner

def gaussian(x,x0,sigma):
  return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power((x - x0)/sigma, 2.)/2.)
  #return np.exp(-np.power((x - x0)/sigma, 2.)/2.)

def multivariate_gaussian(x, mean, cov):
  n = mean.shape[0]
  cov_det = np.linalg.det(cov)
  cov_inv = np.linalg.inv(cov)
  N = np.sqrt((2*np.pi)**n * cov_det)
  # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
  # way across all the input variables.
  fac = np.einsum('...k,kl,...l->...', x-mean, cov_inv, x-mean)
  return np.exp(-fac / 2) / N

  
def gaussian_contour_2d(ax2d, mean, cov, levels):
  xmin, xmax = ax2d.get_xlim()
  ymin, ymax = ax2d.get_ylim()
  x_array = np.linspace(xmin, xmax, 1000)
  y_array = np.linspace(ymin, ymax, 1000)
  xy_array = np.meshgrid(x_array, y_array)
  xy_flat = np.vstack((xy_array[0].flatten(), xy_array[1].flatten()))
  z_array = multivariate_gaussian(np.swapaxes(xy_flat, 0, 1), mean, cov).reshape((x_array.size, y_array.size))

  # the contour plot:
  n = 1000
  z_array = z_array/z_array.sum()
  t = np.linspace(0, z_array.max(), n)
  integral = ((z_array >= t[:, None, None]) * z_array).sum(axis=(1,2))
  
  f = interpolate.interp1d(integral, t)
  t_contours = f(np.array(levels))
  #ax2d.contour(xy_array[1], xy_array[0], z_array, t_contours, linewidths=2.0)
  return xy_array, z_array, t_contours


nside = 32
#infile = "/home/clement/Documents/Projets/Physique/Postdoc_APC/Software/fgbuster/scripts/examples/data/noise_CMB_fullrand_fsky.txt"
infile = "/home/clement/Documents/Projets/Physique/Postdoc_APC/Software/fgbuster/scripts/examples/data/noise_CMB_so_nomask.txt"

# Define sky configuration
sky = pysm.Sky(get_sky(nside, 'c1d0')) # a la fgbuster
instrument = pysm.Instrument(get_instrument('so_la', nside))
#instrument = pysm.Instrument(get_instrument('semiblind_test', nside))
freq_maps = instrument.observe(sky, write_outputs=False)[0]

# Define what you fit for
components = [CMB(), Dust(150.)]
nblind = len(components)-1

#Inverse noise matrix
bl = [hp.gauss_beam(np.radians(b/60.), lmax=3*nside-1) for b in instrument.Beams]
inv_Nl = (np.array(bl) / np.radians(instrument.Sens_P/60.)[:, np.newaxis])**2
inv_Nl = np.array([np.diag(inv_Nl[:,l]) for l in np.arange(inv_Nl.shape[1])])
inv_N = np.diag(hp.nside2resol(nside, arcmin=True) / (instrument.Sens_P))**2
    
#CMB prior covariance matrix
#templates = '/mnt/PersoPro/Documents/Projets/Physique/Postdoc_APC/Software/fgbuster/fgbuster/templates/Cls_Planck2018_lensed_scalar.fits'
templates = np.swapaxes(np.genfromtxt("../CAMB_Nov13/fgbuster/test_lenspotentialCls.dat", usecols=(1, 2, 3)), 0, 1)

#L_brute, L_param, L_ana = test_fisher(components, instrument, templates, freq_maps, nside, nblind, inv_Nl)
fisher = test_fisher(components, instrument, templates, freq_maps, nside, nblind, inv_Nl)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.weight"] = "light"
plt.rc('text', usetex=True)

x0 = [ 0.56436459, 4.26891404, 10.67339696, 36.41225095, 82.60680759]
#x0 = [0.45839486, 2.34451664]
lab = [r'$A_{01}$', r'$A_{21}$', r'$A_{31}$', r'$A_{41}$', r'$A_{51}$']
names = ['$A0$', '$A2$', '$A3$', '$A4$', '$A5$']

'''
plt.figure(1)
plt.imshow(np.log(np.abs(fisher)))
plt.xticks(np.arange(len(lab)), lab, fontsize=17)
plt.yticks(np.arange(len(lab)), lab, fontsize=17)
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$\mathrm{ln} \left| I_{fisher} \right|$', fontsize=20, rotation=270, labelpad=35)
'''

x = np.genfromtxt(infile)
levels = [0.95, 0.66]
#x_gd = MCSamples(samples=x, names=names, labels=lab)
#g=plots.GetDistPlotter()
#g.settings.norm_1d_density=True
#g.triangle_plot(x_gd, filled=True)
figure = corner(x, hist_kwargs={'density':True}, levels=levels)

# Extract the axes
#axes = np.array(g.subplots).reshape((len(x0), len(x0)))
axes = np.array(figure.axes).reshape((len(x0), len(x0)))
cov = np.linalg.inv(fisher)
mean = np.zeros(len(x0))

# Loop over the diagonal
for i in range(len(x0)):
    ax = axes[i, i]
    y_hist, x_hist = np.histogram(x[:, i], bins=20)
    x_fish = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
    ax.axvline(x0[i], color="crimson", linestyle='--')
    mean[i] = norm.fit(x[:, i], scale=np.sqrt(cov[i, i]))[0]
    ax.plot(x_fish, gaussian(x_fish, mean[i], np.sqrt(cov[i, i])), 'k')
    
# Loop over the histograms
for yi in range(len(x0)):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(x0[xi], color="crimson", linestyle='--')
        ax.axhline(x0[yi], color="crimson", linestyle='--')
        ax.plot(x0[xi], x0[yi], "s", color='crimson')
        xy_array, z_array, t_contours = gaussian_contour_2d(ax, np.array((mean[xi], mean[yi])), np.array(((cov[xi, xi], cov[xi, yi]), (cov[yi, xi], cov[yi, yi]))), levels)
        ax.contour(xy_array[0], xy_array[1], z_array, t_contours, linewidths=2.0, colors=['k', 'k'])

#fig.savefig('scripts/examples/plots/noise+CMB_so_la.png')
plt.show()
