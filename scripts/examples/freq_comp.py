# Example from doc
# Simple component separation

import healpy as hp
import pysm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.weight"] = "light"
plt.rc('text', usetex=True)

n=20
n2 = 20

x1, y1, result1 = np.genfromtxt("scripts/examples/data/grid_test_S_39_145_200.txt")
x2, y2, result2 = np.genfromtxt("scripts/examples/data/grid_test_S_75_145_200.txt")
x3, y3, result3 = np.genfromtxt("scripts/examples/data/grid_test_S_100_145_200.txt")
x4, y4, result4 = np.genfromtxt("scripts/examples/data/grid_test_S_39_145_280.txt")
x5, y5, result5 = np.genfromtxt("scripts/examples/data/grid_test_S_75_145_280.txt")
x6, y6, result6 = np.genfromtxt("scripts/examples/data/grid_test_S_100_145_280.txt")
x7, y7, result7 = np.genfromtxt("scripts/examples/data/grid_test_S_39_145_350.txt")
x8, y8, result8 = np.genfromtxt("scripts/examples/data/grid_test_S_75_145_350.txt")
x9, y9, result9 = np.genfromtxt("scripts/examples/data/grid_test_S_100_145_350.txt")

x1 = x1.reshape((n, n))
y1 = y1.reshape((n, n))
result1 = result1.reshape((n, n))
x2 = x2.reshape((n, n))
y2 = y2.reshape((n, n))
result2 = result2.reshape((n, n))
x3 = x3.reshape((n, n))
y3 = y3.reshape((n, n))
result3 = result3.reshape((n, n))
x4 = x4.reshape((n, n))
y4 = y4.reshape((n, n))
result4 = result4.reshape((n, n))
x5 = x5.reshape((n, n))
y5 = y5.reshape((n, n))
result5 = result5.reshape((n, n))
x6 = x6.reshape((n, n))
y6 = y6.reshape((n, n))
result6 = result6.reshape((n, n))
x7 = x7.reshape((n, n))
y7 = y7.reshape((n, n))
result7 = result7.reshape((n, n))
x8 = x8.reshape((n, n))
y8 = y8.reshape((n, n))
result8 = result8.reshape((n, n))
x9 = x9.reshape((n, n))
y9 = y9.reshape((n, n))
result9 = result9.reshape((n, n))


levels1 = MaxNLocator(nbins=50).tick_values(result1.min(), result1.max())
ind1 = np.unravel_index(np.argmax(result1, axis=None), result1.shape)
levels2 = MaxNLocator(nbins=50).tick_values(result2.min(), result2.max())
ind2 = np.unravel_index(np.argmax(result2, axis=None), result2.shape)
levels3 = MaxNLocator(nbins=50).tick_values(result3.min(), result3.max())
ind3 = np.unravel_index(np.argmax(result3, axis=None), result3.shape)
levels4 = MaxNLocator(nbins=50).tick_values(result4.min(), result4.max())
ind4 = np.unravel_index(np.argmax(result4, axis=None), result4.shape)
levels5 = MaxNLocator(nbins=50).tick_values(result5.min(), result5.max())
ind5 = np.unravel_index(np.argmax(result5, axis=None), result5.shape)
levels6 = MaxNLocator(nbins=50).tick_values(result6.min(), result6.max())
ind6 = np.unravel_index(np.argmax(result6, axis=None), result6.shape)
levels7 = MaxNLocator(nbins=50).tick_values(result7.min(), result7.max())
ind7 = np.unravel_index(np.argmax(result7, axis=None), result7.shape)
levels8 = MaxNLocator(nbins=50).tick_values(result8.min(), result8.max())
ind8 = np.unravel_index(np.argmax(result8, axis=None), result8.shape)
levels9 = MaxNLocator(nbins=50).tick_values(result9.min(), result9.max())
ind9 = np.unravel_index(np.argmax(result9, axis=None), result9.shape)

cmap = plt.get_cmap('CMRmap')
norm1 = BoundaryNorm(levels1, ncolors=cmap.N, clip=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap.N, clip=True)
norm3 = BoundaryNorm(levels3, ncolors=cmap.N, clip=True)
norm4 = BoundaryNorm(levels4, ncolors=cmap.N, clip=True)
norm5 = BoundaryNorm(levels5, ncolors=cmap.N, clip=True)
norm6 = BoundaryNorm(levels6, ncolors=cmap.N, clip=True)
norm7 = BoundaryNorm(levels7, ncolors=cmap.N, clip=True)
norm8 = BoundaryNorm(levels8, ncolors=cmap.N, clip=True)
norm9 = BoundaryNorm(levels9, ncolors=cmap.N, clip=True)

#fig = plt.figure(figsize=(12 ,12))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(15, 10), nrows=3, ncols=3)

cf1 = ax1.contourf(x1,
                  y1, result1, levels=levels1,
                  cmap=cmap)
cbar1 = fig.colorbar(cf1, ax=ax1)
ax1.plot(x1[ind1], y1[ind1], '+')
ax1.plot([x1.min(), x1.max()], [0, 0], 'gray', linestyle='--')
ax1.plot([0, 0], [y1.min(), y1.max()], 'gray', linestyle='--')
ax1.set_title('(39, 145, 200)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax1.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax1.set_title('Pixel', fontsize=25)
#ax1.set_xlabel(r'$A_{01}$', fontsize=20)
#ax1.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar1.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax1.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar1.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf2 = ax2.contourf(x2,
                  y2, result2, levels=levels2,
                  cmap=cmap)
cbar2 = fig.colorbar(cf2, ax=ax2)
ax2.plot(x2[ind1], y2[ind1], '+')
ax2.plot([x2.min(), x2.max()], [0, 0], 'gray', linestyle='--')
ax2.plot([0, 0], [y2.min(), y2.max()], 'gray', linestyle='--')
ax2.set_title('(75, 145, 200)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax2.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax2.set_title('Pixel', fontsize=25)
#ax2.set_xlabel(r'$A_{01}$', fontsize=20)
#ax2.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar2.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax2.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar2.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf3 = ax3.contourf(x3,
                  y3, result3, levels=levels3,
                  cmap=cmap)
cbar3 = fig.colorbar(cf3, ax=ax3)
ax3.plot(x3[ind1], y3[ind1], '+')
ax3.plot([x3.min(), x3.max()], [0, 0], 'gray', linestyle='--')
ax3.plot([0, 0], [y3.min(), y3.max()], 'gray', linestyle='--')
ax3.set_title('(100, 145, 200)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax3.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax3.set_title('Pixel', fontsize=25)
#ax3.set_xlabel(r'$A_{01}$', fontsize=20)
#ax3.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar3.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax3.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar3.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf4 = ax4.contourf(x4,
                  y4, result4, levels=levels4,
                  cmap=cmap)
cbar4 = fig.colorbar(cf4, ax=ax4)
ax4.plot(x4[ind1], y4[ind1], '+')
ax4.plot([x4.min(), x4.max()], [0, 0], 'gray', linestyle='--')
ax4.plot([0, 0], [y4.min(), y4.max()], 'gray', linestyle='--')
ax4.set_title('(39, 145, 280)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax4.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax4.set_title('Pixel', fontsize=25)
#ax4.set_xlabel(r'$A_{01}$', fontsize=20)
#ax4.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar4.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax4.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar4.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf5 = ax5.contourf(x5,
                  y5, result5, levels=levels5,
                  cmap=cmap)
cbar5 = fig.colorbar(cf5, ax=ax5)
ax5.plot(x5[ind1], y5[ind1], '+')
ax5.plot([x5.min(), x5.max()], [0, 0], 'gray', linestyle='--')
ax5.plot([0, 0], [y5.min(), y5.max()], 'gray', linestyle='--')
ax5.set_title('(75, 145, 280)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax5.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax5.set_title('Pixel', fontsize=25)
#ax5.set_xlabel(r'$A_{01}$', fontsize=20)
#ax5.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar5.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax5.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar5.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf6 = ax6.contourf(x6,
                  y6, result6, levels=levels6,
                  cmap=cmap)
cbar6 = fig.colorbar(cf6, ax=ax6)
ax6.plot(x6[ind1], y6[ind1], '+')
ax6.plot([x6.min(), x6.max()], [0, 0], 'gray', linestyle='--')
ax6.plot([0, 0], [y6.min(), y6.max()], 'gray', linestyle='--')
ax6.set_title('(100, 145, 280)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax6.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax6.set_title('Pixel', fontsize=25)
#ax6.set_xlabel(r'$A_{01}$', fontsize=20)
#ax6.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar6.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax6.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar6.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf7 = ax7.contourf(x7,
                  y7, result7, levels=levels7,
                  cmap=cmap)
cbar7 = fig.colorbar(cf7, ax=ax7)
ax7.plot(x7[ind1], y7[ind1], '+')
ax7.plot([x7.min(), x7.max()], [0, 0], 'gray', linestyle='--')
ax7.plot([0, 0], [y7.min(), y7.max()], 'gray', linestyle='--')
ax7.set_title('(39, 145, 350)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax7.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax7.set_title('Pixel', fontsize=25)
#ax7.set_xlabel(r'$A_{01}$', fontsize=20)
#ax7.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar7.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax7.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar7.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf8 = ax8.contourf(x8,
                  y8, result8, levels=levels8,
                  cmap=cmap)
cbar8 = fig.colorbar(cf8, ax=ax8)
ax8.plot(x8[ind1], y8[ind1], '+')
ax8.plot([x8.min(), x8.max()], [0, 0], 'gray', linestyle='--')
ax8.plot([0, 0], [y8.min(), y8.max()], 'gray', linestyle='--')
ax8.set_title('(75, 145, 350)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax8.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax8.set_title('Pixel', fontsize=25)
#ax8.set_xlabel(r'$A_{01}$', fontsize=20)
#ax8.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar8.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax8.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar8.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

cf9 = ax9.contourf(x9,
                  y9, result9, levels=levels9,
                  cmap=cmap)
cbar9 = fig.colorbar(cf9, ax=ax9)
ax9.plot(x9[ind1], y9[ind1], '+')
ax9.plot([x9.min(), x9.max()], [0, 0], 'gray', linestyle='--')
ax9.plot([0, 0], [y9.min(), y9.max()], 'gray', linestyle='--')
ax9.set_title('(100, 145, 350)', fontsize=25)
#print('max noS : ({}, {})'.format(x[indS], y[indS]))
#ax9.set_title(r'$S^{-1} = 0$', fontsize=25)
#ax9.set_title('Pixel', fontsize=25)
#ax9.set_xlabel(r'$A_{01}$', fontsize=20)
#ax9.set_ylabel(r'$A_{21}$', fontsize=20)
#cbar9.ax.set_ylabel(r'$\mathrm{ln} \mathcal{L}$', rotation=270, fontsize=20, labelpad = 25)
ax9.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)
cbar9.ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=1.5)

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout(h_pad=2)

fig.savefig('scripts/examples/plots/S_freq_comp.png')
#fig.savefig('scripts/examples/plots/3x2_SnoS.png')
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
