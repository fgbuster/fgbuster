import pysm
from fgbuster.observation_helpers import get_instrument, get_sky
import healpy as hp
import numpy as np
from fgbuster import xForecast, CMB, Dust, Synchrotron

nside = 64
# define sky and foregrounds simulations
sky = pysm.Sky(get_sky(nside, 'd0s0')) # why no CMB ?
# define instrument
instrument = pysm.Instrument(get_instrument('cmbs4', nside))
# get noiseless frequency maps
freq_maps = instrument.observe(sky, write_outputs=False)[0] # why noiseless ?
# take only the Q and U maps
freq_maps = freq_maps[:,1:]


# create 3% circular sky mask
RA = 2*np.pi-70.*np.pi/180
DEC = np.pi/2+70.*np.pi/180
radius = 34*np.pi/180
mask_circular = np.zeros(12*nside**2)
for ipix in range(12*nside**2):
    theta, phi = hp.pix2ang(nside, ipix)
    if (((phi - RA)**2 + (theta - DEC)**2 <= radius**2)):
        mask_circular[ipix] = 1.0
    if (((phi - RA+2*np.pi)**2 + (theta - DEC)**2 <= radius**2)):
        mask_circular[ipix] = 1.0
# applying mask to observed frequency maps
freq_maps[...,mask_circular==0] = 0.0


# masking for litebird
mask = hp.read_map('/Users/josquin1/Documents/Dropbox/CNRS-CR2/CMBX4cast/HFI_Mask_GalPlane-apo2_2048_R2.00.fits', field=(2))
mask = hp.ud_grade(mask, nside_out=nside)
#mask_patch = hp.ud_grade(mask, nside_out=nside_patch)
#mask_patch = hp.ud_grade(mask_patch, nside_out=NSIDE)
# hp.mollview(mask_patch)
# pl.show()
#mask_patch_bin = mask_patch*0.0
#mask_patch_bin[np.where(mask_patch!=0)[0]] = 1.0
mask = np.ones(mask.shape)
freq_maps[:,:,np.where(mask==0)[0]] = 0.0
#fsky = len(np.where(mask_patch!=0.0)[0])*1.0/len(mask_patch)
#print('fsky = ', fsky)
# fsky_ = len(np.where(mask!=0.0)[0])*1.0/len(mask)
# print('fsky_ = ', fsky_)
#fsky_patch = len(np.where(mask_patch_bin!=0.0)[0])*1.0/len(mask_patch_bin)

# define components used in the modeling
components = [CMB(), Dust(150., temp=19.6), Synchrotron(150.)]
#components = [CMB(), Dust(150.), Synchrotron(150.)]


# call for xForecast
# with lmin=2, lmax=2*nside-1, and Alens=0.1
# you can try with make_figure=True if you want to output angular power spectra and profile likelihood on r
res = xForecast(components, instrument, freq_maps, 2, 2*nside-1, Alens=0.1, r=0.001, make_figure=True)
