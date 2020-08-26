import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy import interpolate

def gaussian(x,x0,sigma):
    #return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power((x - x0)/sigma, 2.)/2.)
    return np.exp(-np.power((x - x0)/sigma, 2.)/2.)

def multivariate_gaussian(x, mean, cov):
    n = mean.shape[0]
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    N = np.sqrt((2*np.pi)**n * cov_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', x-mean, cov_inv, x-mean)
    return np.exp(-fac/2)/N

def gaussian_contour_2d(ax2d, mean, cov, levels, colors=['k', 'k']):
    #xmin, xmax = ax2d.get_xlim()
    #ymin, ymax = ax2d.get_ylim()
    xmin = mean[0]-4*np.sqrt(cov[0, 0])
    xmax = mean[0]+4*np.sqrt(cov[0, 0])
    ymin = mean[1]-4*np.sqrt(cov[1, 1])
    ymax = mean[1]+4*np.sqrt(cov[1, 1])

    x_array = np.linspace(xmin, xmax, 100)
    y_array = np.linspace(ymin, ymax, 100)
    xy_array = np.meshgrid(x_array, y_array)
    xy_flat = np.vstack((xy_array[0].flatten(), xy_array[1].flatten()))
    z_array = multivariate_gaussian(np.swapaxes(xy_flat, 0, 1), mean, cov).reshape((x_array.size, y_array.size))

    # the contour plot:
    n = 100
    z_array = z_array/z_array.sum()
    t = np.linspace(0, z_array.max(), n)
    integral = ((z_array >= t[:, None, None]) * z_array).sum(axis=(1,2))

    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array(levels))
    ax2d.contour(xy_array[0], xy_array[1], z_array, t_contours, linewidths=2.0, colors=colors)

def corner_plot(mean, cov, levels=[0.95, 0.66], params=None, truth=None, colors=None, legend=False, labels=None):

    assert len(mean.shape) <= 2 #check that mean is at most 2-dimensional
    assert len(cov.shape) == len(mean.shape) + 1

    if len(mean.shape) == 1:
        mean = mean[np.newaxis, :]
        cov = cov[np.newaxis, ...]

    nsample = mean.shape[0]
    dim = mean.shape[1]
  
    assert cov.shape[-2:] == (dim, dim) #check that cov has right shape

    # Handle parameter names
    if params is None:
        params = np.asarray(["param_" + str(i) for i in np.arange(dim)])
    assert len(params) == dim

    # Handle legend labels
    if legend == True:
        assert len(labels) == nsample
    else:
        labels = np.full(nsample, '')
        

    factor = 3.0
    whspace = 0.05
    plotdim = factor*dim + factor*dim*whspace
    fig, axes = plt.subplots(dim, dim, figsize=(plotdim, plotdim))
    
    #Format figure
    fig.subplots_adjust(wspace=whspace, hspace=whspace)


    # Handle colors
    if colors is None:
        colors = np.asarray(["C" + str(i) for i in np.arange(nsample)])
    assert len(colors) == mean.shape[0]
    #color1 = np.array(col.to_rgb('darkslateblue'))
    colors2d = np.asarray([[np.minimum(np.array(col.to_rgb(colors[i])) + 0.2*j, np.ones(3)) for j in np.arange(len(levels))] for i in np.arange(len(colors))])
    print(colors2d.shape)

    # Loop over the diagonal
    for i in np.arange(dim):
        ax = axes[i, i]

        if i != 0:
            ax.set_yticks([])

        if i != dim-1:
            ax.set_xticks([])

        ax.set_xlim(np.min(mean[:, i]-4*np.sqrt(cov[:, i, i])), np.max(mean[:, i]+4*np.sqrt(cov[:, i, i])))
        ax.set_ylim(0, 1.25)

        for j in np.arange(nsample):
            x1d = np.linspace(mean[j, i]-4*np.sqrt(cov[j, i, i]), mean[j, i]+4*np.sqrt(cov[j, i, i]), 100)
            if i == 0:
                ax.plot(x1d, gaussian(x1d, mean[j, i], np.sqrt(cov[j, i, i])), color=colors[j], label=labels[j])
            else:
                ax.plot(x1d, gaussian(x1d, mean[j, i], np.sqrt(cov[j, i, i])), color=colors[j])

        if truth is not None:
            ax.axvline(truth[i], color="crimson", linestyle='--')

    # Loop over the 2d contours
    for yi in np.arange(dim):
        for xi in np.arange(dim):
            ax = axes[yi, xi]

            if xi != 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(params[yi])

            if yi != dim-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(params[xi])
                
            if yi<xi:
                ax.set_frame_on(False)
            elif yi>xi:
                ax.set_xlim(np.min(mean[:, xi]-4*np.sqrt(cov[:, xi, xi])), np.max(mean[:, xi]+4*np.sqrt(cov[:, xi, xi])))
                ax.set_ylim(np.min(mean[:, yi]-4*np.sqrt(cov[:, yi, yi])), np.max(mean[:, yi]+4*np.sqrt(cov[:, yi, yi])))
                for j in np.arange(nsample):
                    gaussian_contour_2d(ax, np.array((mean[j, xi], mean[j, yi])), np.array(((cov[j, xi, xi], cov[j, xi, yi]), (cov[j, yi, xi], cov[j, yi, yi]))), levels, colors=colors2d[j,::-1,:])

                if truth is not None:
                    ax.plot(truth[xi], truth[yi], "s", color='crimson')
                    ax.axvline(truth[xi], color="crimson", linestyle='--')
                    ax.axhline(truth[yi], color="crimson", linestyle='--')

    if legend==True:
        fig.legend(loc='upper right')

    return fig

def plot1d(x, y, params=None, truth=None, colors=None, legend=False, labels=None, xlogscale=False, ylogscale=False, fig_in=None, **plot_kwargs):
    
    assert x.shape == y.shape #check that x and y have same size

    if len(x.shape) == 1:
        x = x[np.newaxis, :]
        y = y[np.newaxis, :]

    nsample = x.shape[0]
  
    # Handle parameter names
    if params is None:
        params = np.asarray(["param_1", "param_2"])

    # Handle legend labels
    if legend == True:
        assert len(labels) == nsample
    else:
        labels = np.full(nsample, '')
        
    if fig_in is None:
        size = 7.0
        whspace = 0.05
        plotdim = size + size*whspace
        fig, ax = plt.subplots(figsize=(plotdim, plotdim))
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), 1.25*np.max(y))
        ax.set_xlabel(params[0], fontsize=20)
        ax.set_ylabel(params[1], fontsize=20)
        if xlogscale:
            ax.set_xscale('log')
        if ylogscale:
            ax.set_yscale('log')
    else:
        fig = fig_in
        ax = fig.axes[0]
        xmin, xmax = np.minimum(ax.get_xlim()[0], np.min(x)), np.maximum(ax.get_xlim()[1], np.max(x))
        ymin, ymax = np.minimum(ax.get_ylim()[0], np.min(y)), np.maximum(ax.get_ylim()[1], np.max(y))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, 1.25*ymax)
    
    # Handle colors
    if colors is None:
        colors = np.asarray(["C" + str(i) for i in np.arange(nsample)])
    assert len(colors) == nsample

    for i in np.arange(nsample):
        ax.plot(x[i,:],  y[i,:], color=colors[i], linewidth=2.0, label=labels[i], **plot_kwargs)
            
    if truth is not None:
        ax.axvline(truth, color="crimson", linestyle='--', label="truth")

    if legend==True:
        ax.legend(loc='upper right')
    
    return fig
    
'''
truth = np.array([1.54, 20., -3.])
mean = np.array([[1.54, 20., -3.], [1.96, 19.7, -2.99]])
cov = np.array([[[ 1.29261499e-05, -4.44505727e-04, -6.49300081e-06],
                [-4.44505727e-04, 1.54017041e-02, 2.10390110e-04],
                 [-6.49300081e-06, 2.10390110e-04, 2.23398702e-05]],
                [[ 1.29261499e-05, -4.44505727e-04, -6.49300081e-06],
                [-4.44505727e-04, 1.54017041e-02, 2.10390110e-04],
                 [-6.49300081e-06, 2.10390110e-04, 2.23398702e-05]]])
params = [r'$\beta_d$', r'$T_d$', r'$\beta_s$']
labels = ['number 1', 'number 2']

corner_plot(mean, cov, params=params, truth=truth, colors=['darkslateblue', 'forestgreen'], legend=True, labels=labels)

plt.show()
'''
