""" All the routines for making all Josquin's lovely plots
"""
from corner import corner
import numpy as np
import matplotlib.pyplot as plt

def corner_norm(mean, cov, *args, **kwargs):
    ''' Corner plot for multivariate gaussian

    Just like corner.corner, but you privide mean and covariance instead of `xs`
    '''
    xs = np.random.multivariate_normal(mean, cov, 100000)  # TODO: not hardcoded
    corner(xs, *args, **kwargs)


def plot_component(component, nu_min, nu_max):
    nus = np.logspace(np.log10(nu_min), np.log10(nu_max), 1000)
    emission = component.eval(nus, *(component.defaults))
    plt.loglog(nus, emission, label=type(component).__name__)
