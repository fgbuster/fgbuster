# FGBuster
# Copyright (C) 2019 Davide Poletti, Josquin Errard and the FGBuster developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

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
