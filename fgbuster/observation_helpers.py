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

""" Provide handy access to pysm ojects
"""
import sys
import numpy as np
import pysm
import healpy as hp


__all__ = [
    'get_sky',
    'get_instrument',
]


def get_sky(nside, tag='c1d0s0'):
    """ Get a pre-defined PySM sky

    Parameters
    ----------
    nside: int
        healpix nside of the sky templates
    tag: string
        See `pysm.nominal
        <https://github.com/bthorne93/PySM_public/blob/master/pysm/nominal.py>`_ 
        for a complete list of available options.
        Default is 'c1d0s0', i.e. cmb (c1), dust with constant temperature and
        spectral index (d0), and synchrotron with constant spectral index (s0).

    Returns
    -------
    sky: dict
        Configuration dictionary. It can be used for constructing a ``PySM.Sky``
    """
    comp_names = {
        'a': 'ame',
        'c': 'cmb',
        'd': 'dust',
        'f': 'freefree',
        's': 'synchrotron',
    }
    sky_config = {}
    for i in range(0, len(tag), 2):
        sky_config[comp_names[tag[i]]] = pysm.nominal.models(tag[i:i+2], nside)

    return sky_config


def get_instrument(tag, nside=None, units='uK_CMB'):
    """ Get a pre-defined instrumental configuration

    Parameters
    ----------
    tag: string
        name of the pre-defined experimental configurations. See the source or
        set tag to something random to have a list of the available
        configurations. It can contain the name of multiple experiments
        separated by a space.
    nside: int
        If you plan to build 

    Returns
    -------
    instr: dict
        It contains the experimetnal configuration of the desired instrument.
        It can be used to construct a ``pysm.Instrument`` or as the
        ``instrument`` argument for, e.g., :func:`basic_comp_sep` and
        :func:`xForecast`
    """
    module = sys.modules[__name__]
    instruments = []
    for t in tag.split():
        try:
            instrument = getattr(module, '_dict_instrument_'+t)(nside, units)
        except AttributeError:
            raise ValueError('Instrument %s not available. Choose between: %s.'
                             % (t, ', '.join(_get_available_instruments())))
        else:
            instruments.append(instrument)

    for key in instruments[0]:
        if isinstance(instruments[0][key], np.ndarray):
            instruments[0][key] = np.concatenate([i[key] for i in instruments])

    instruments[0]['prefix'] = '__'.join(tag.split())

    return instruments[0]


def _get_available_instruments():
    prefix = '_dict_instrument_'
    module = sys.modules[__name__]
    has_prefix = lambda x: x.find(prefix) != -1
    return [fun.replace(prefix, '') for fun in dir(module) if has_prefix(fun)]


def _dict_instrument_test(nside=None, units='uK_CMB'):
    # Mock instrument configuration for testing purposes
    return {
        'frequencies': np.arange(10., 300, 30.),
        'sens_I': (np.linspace(20, 40, 10) - 30)**2,
        'sens_P': (np.linspace(20, 40, 10) - 30)**2,
        'beams': np.ones(9),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'test',
        'use_smoothing': False,
    }


def _dict_instrument_planck_P(nside=None, units='uK_CMB'):
    # Planck 2015 results X, A&A, Volume 594, id.A10, 63 pp.
    return {
        'frequencies': np.array([30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0]),
        'sens_I': np.array([2.8, 3.0, 4.0, 0.85, 0.9, 1.8, 4]) * 60,
        'sens_P': np.array([7.5, 7.5, 4.8, 1.3, 1.1, 1.6, 6.9]) * 40.0,
        'beams': np.array([32.4, 27.1, 13.3, 9.5, 7.2, 5.0, 4.9]),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'planck',
        'use_smoothing': False,
    }


def _dict_instrument_litebird(nside=None, units='uK_CMB'):
    # Matsumura et al., Journal of Low Temperature Physics 184, 824 (2016)
    return {
        'frequencies': np.array([40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9, 140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1]),
        'sens_I': np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6]) / 1.41,
        'sens_P': np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6]),
        'beams': np.array([60, 56, 48, 43, 39, 35, 29, 25, 23, 21, 20, 19, 24, 20, 17]),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'litebird',
        'use_smoothing': False,
    }


def _dict_instrument_pico(nside=None, units='uK_CMB'):
    # https://sites.google.com/umn.edu/picomission/home
    return {
        'frequencies': np.array([21.0, 25.0, 30.0, 36.0, 43.2, 51.8, 62.2, 74.6, 89.6, 107.5, 129.0, 154.8, 185.8, 222.9, 267.5, 321.0, 385.2, 462.2, 554.7, 665.6, 798.7]),
        'sens_I': np.array([16.9, 11.8, 8.1, 5.7, 5.8, 4.1, 3.8, 2.9, 2.0, 1.6, 1.6, 1.3, 2.6, 3.0, 2.1, 2.9, 3.5, 7.4, 34.6, 143.7, 896.4]) / 1.41,
        'sens_P': np.array([16.9, 11.8, 8.1, 5.7, 5.8, 4.1, 3.8, 2.9, 2.0, 1.6, 1.6, 1.3, 2.6, 3.0, 2.1, 2.9, 3.5, 7.4, 34.6, 143.7, 896.4]),
        'beams': np.array([40.9, 34.1, 28.4, 23.7, 19.7, 16.4, 13.7, 11.4, 9.5, 7.9, 6.6, 5.5, 4.6, 3.8, 3.2, 2.7, 2.2, 1.8, 1.5, 1.3, 1.1]),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'pico',
        'use_smoothing': False,
    }


def _dict_instrument_cmbs4(nside=None, units='uK_CMB'):
    # https://cmb-s4.org/wiki/index.php/Survey_Performance_Expectations
    return {
        'frequencies': np.array([20, 30, 40, 85, 95, 145, 155, 220, 270]),
        'sens_I': np.array([16.66, 10.62, 10.07, 2.01, 1.59, 4.53, 4.53, 11.61, 15.84]),
        'sens_P': np.array([13.6, 8.67, 8.22, 1.64, 1.30, 2.03, 2.03, 5.19, 7.08]),
        'beams': np.array([11.0, 76.6, 57.5, 27.0, 24.2, 15.9, 14.8, 10.7, 8.5]),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'cmbs4',
        'use_smoothing': False,
    }
