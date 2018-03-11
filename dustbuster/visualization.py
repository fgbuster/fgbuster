""" All the routines for making all Josquin's lovely plots
"""
from corner import corner
import numpy as np

def corner_norm(mean, cov, *args, **kwargs):
    ''' Corner plot for multivariate gaussian

    Just like corner.corner, but you privide mean and covariance instead of `xs`
    '''
    xs = np.random.multivariate_normal(mean, cov, 100000)  # TODO: not hardcoded 
    corner(xs, *args, **kwargs)
