from __future__ import print_function
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
from getdist import plots, MCSamples
import getdist
# use this *after* importing getdist if you want to use interactive plots
# %matplotlib notebook
import matplotlib.pyplot as plt
import IPython
print('GetDist Version: %s, Matplotlib version: %s'%(getdist.__version__, plt.matplotlib.__version__))
import numpy as np

ndim = 5
nsamp = 10000
np.random.seed(10)
A = np.random.rand(ndim,ndim)
cov = np.dot(A, A.T)
samps = np.random.multivariate_normal([0]*ndim, cov, size=nsamp)
A = np.random.rand(ndim,ndim)
cov = np.dot(A, A.T)
samps2 = np.random.multivariate_normal([0]*ndim, cov, size=nsamp)

x = np.genfromtxt("/home/clement/Documents/Projets/Physique/Postdoc_APC/Software/fgbuster/scripts/examples/data/noise_CMB_so.txt")


names = ["x%s"%i for i in range(ndim)]
labels =  [r"A_%s"%i for i in range(ndim)]
samples = MCSamples(samples=samps,names = names, labels = labels)
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Second set')
samples3 = MCSamples(samples=x, names=names, labels=labels, label='Third set')

g = plots.GetDistPlotter()
g.triangle_plot([samples3], filled=True)

g.export(os.path.join(r'./',r'test.png'))
