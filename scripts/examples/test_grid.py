# Example from doc
# Simple component separation

import healpy as hp
import pysm
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.weight"] = "light"
plt.rc('text', usetex=True)

n=20
n2 = 50
#n2 = 20

#x1, y1, result1 = np.genfromtxt("scripts/examples/data/S_100_145_200_noise.txt")
#x2, y2, result2 = np.genfromtxt("scripts/examples/data/S_100_145_200_noise_zoom.txt")
#x1, y1, result1 = np.genfromtxt("scripts/examples/data/S_noE_mask_noise.txt")
#x2, y2, result2 = np.genfromtxt("scripts/examples/data/S_noE_mask_noise_zoom.txt")
#x1, y1, result1 = np.genfromtxt("scripts/examples/data/S_mask_lmin30_noise.txt")
#x2, y2, result2 = np.genfromtxt("scripts/examples/data/S_mask_lmin30_noise_zoom.txt")
#x1, y1, result1 = np.genfromtxt("scripts/examples/data/noS_full_pix.txt")
#x2, y2, result2 = np.genfromtxt("scripts/examples/data/noS_full_zoom_pix.txt")
x1, y1, result1 = np.genfromtxt("scripts/examples/data/trueS_full_noise.txt")
x2, y2, result2 = np.genfromtxt("scripts/examples/data/trueS_full_noise_zoom.txt")
#x1, y1, result1 = np.genfromtxt("scripts/examples/data/trueS_mask_noise.txt")
#x2, y2, result2 = np.genfromtxt("scripts/examples/data/trueS_mask_noise_zoom.txt")
#x1, y1, result1 = np.genfromtxt("scripts/examples/data/trueS_mask_lmin30_noise.txt")
#x2, y2, result2 = np.genfromtxt("scripts/examples/data/trueS_mask_lmin30_noise_zoom.txt")

x1 = x1.reshape((n, n))
y1 = y1.reshape((n, n))
result1 = result1.reshape((n, n))
x2 = x2.reshape((n2, n2))
y2 = y2.reshape((n2, n2))
result2 = result2.reshape((n2, n2))


levels1 = MaxNLocator(nbins=50).tick_values(result1.min(), result1.max())
ind1 = np.unravel_index(np.argmax(result1, axis=None), result1.shape)
levels2 = MaxNLocator(nbins=50).tick_values(result2.min(), result2.max())
ind2 = np.unravel_index(np.argmax(result2, axis=None), result2.shape)
levelsc = np.array([result2.max()-4.5, result2.max()-2, result2.max()-0.5])
#levels2 = result2[ind2]*sigmas
#print(levels2)
print('max noS : ({}, {})'.format(x2[ind2], y2[ind2]))

cmap = plt.get_cmap('CMRmap')
norm1 = BoundaryNorm(levels1, ncolors=cmap.N, clip=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap.N, clip=True)

fig, (ax1, ax2) = plt.subplots(figsize=(10 ,10), nrows=2)


#fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(15, 10), nrows=3, ncols=3)

cf1 = ax1.contourf(x1,
                  y1, result1, levels=levels1,
                  cmap=cmap)
cbar1 = fig.colorbar(cf1, ax=ax1)
ax1.plot(x1[ind1], y1[ind1], '+')
ax1.plot(0.45839486, 2.34451664, 'r+')
ax1.plot([x1.min(), x1.max()], [0, 0], 'gray', linestyle='--')
ax1.plot([0, 0], [y1.min(), y1.max()], 'gray', linestyle='--')
#ax1.set_title('Masked galactic plane', fontsize=25, labelpad=25)
#ax1.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax1.set_title('Pixel', fontsize=25)
#ax1.set_xlabel(r'$A_{01}$', fontsize=25, labelpad=15)
ax1.set_ylabel(r'$A_{21}$', fontsize=25, labelpad=10)
cbar1.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=25, labelpad=40)
ax1.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar1.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf2 = ax2.contourf(x2,
                  y2, result2, levels=levels2,
                  cmap=cmap)
cbar2 = fig.colorbar(cf2, ax=ax2)
ax2.plot(x2[ind2], y2[ind2], '+')
ax2.plot(0.45839486, 2.34451664, 'r+')
#ax2.set_title('(100, 145, 200)', fontsize=25)
#ax2.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax2.set_title('Pixel', fontsize=25)
ax2.set_xlabel(r'$A_{01}$', fontsize=25, labelpad = 15)
ax2.set_ylabel(r'$A_{21}$', fontsize=25, labelpad = 10)
cbar2.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=25, labelpad = 40)
ax2.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar2.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
c = ax2.contour(cf2, levels=levelsc, colors='k', linewidths=2.0)
fmt = {}
strs = [r'$3 \sigma$', r'$2 \sigma$', r'$1 \sigma$']
for l, s in zip(c.levels, strs):
    fmt[l] = s
ax2.clabel(c, c.levels, inline=True, fmt=fmt, usetex=True, fontsize=15, weight='Bold')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap

fig.suptitle(r'Full maps, $S^{-1} = S^{-1}_{true} \neq 0$', fontsize=25)
#fig.tight_layout(pad=3.0)
fig.savefig('scripts/examples/plots/trueS_full_noise.png')
#fig.savefig('scripts/examples/plots/S_full_noise.png')
#plt.show()

#Explore the results
#print(result.params)
#print(result.x)

#corner_norm(result.x, result.Sigma, labels=result.params)

#print(result.s.shape)

#hp.mollview(result.s[0,1], title='CMB')
#hp.mollview(result.s[1,1], title='Dust', norm='hist')
#hp.mollview(result.s[2,1], title='Synchrotron', norm='hist')
#plt.show()
