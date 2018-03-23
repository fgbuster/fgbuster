""" Provide handy access to pysm ojects
"""
import sys
import numpy as np
import pysm
import inspect

def get_sky(nside, tag='c1d0s0'):
    """ Get a pre-defined PySM sky

    Parameters
    ----------
    tag: string
        Default is 'c1d0s0', i.e. cmb (c1), dust with constant temperature and
        spectral index (d0), and synchrotron with constant spectral index (s0).
        See pysm.nominal for a complete list of available options.

    Returns
    -------
    sky: PySM.Sky
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

    return pysm.Sky(sky_config)


def get_instrument(nside, tag, units='K_CMB'):
    """ Get pre-defined PySM Instrument

    Parameters
    ----------
    tag: string
        name of the pre-defined experimental configuration. See the source or
        set tag to something random to have a list of the available
        configurations.

    Returns
    -------
    sky: PySM.Instrument
    """
    module = sys.modules[__name__]
    try:
        instrument = getattr(module, '_dict_instrument_'+tag)(nside, units)
    except AttributeError:
        raise ValueError('Instrument %s not available. Chose between: %s.'
                         % (tag, ', '.join(_get_available_instruments())))
    return pysm.Instrument(instrument)


def _get_available_instruments():
    prefix = '_dict_instrument_'
    module = sys.modules[__name__]
    has_prefix = lambda x: x.find(prefix) != -1
    return [fun.replace(prefix, '') for fun in dir(module) if has_prefix(fun)]


def _dict_instrument_test(nside, units='uK_CMB'):
    return {
        'frequencies': np.arange(50., 700, 50.),
        'sens_I': np.ones(13),
        'sens_P': np.ones(13),
        'beams': np.ones(13),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'test',
        'use_smoothing': True,
    }


def _dict_instrument_planck_P(nside, units='uK_CMB'):
    return {
        'frequencies': np.array([30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0]),
        'sens_I': np.array([7.5, 7.5, 4.8, 1.3, 1.1, 1.6, 6.9]) * 40.0, # XXX
        'sens_P': np.array([7.5, 7.5, 4.8, 1.3, 1.1, 1.6, 6.9]) * 40.0,
        'beams': np.array([33.16, 28.09, 13.08, 9.66, 7.27, 5.01, 4.86]),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'planck',
        'use_smoothing': True,
    }


def _dict_instrument_litebird(nside, units='uK_CMB'):
    return {
	'frequencies': np.array([40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9, 140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1]),
        'sens_I': np.array([42.4,  25.8,  20.1,  15.6, 12.5, 10.1,  11.8, 9.5, 7.6,   6.7, 5.1,   6.3, 10.1,  10.1,  19.1]) / 1.41,
        'sens_P': np.array([42.4,  25.8,  20.1,  15.6, 12.5, 10.1,  11.8, 9.5, 7.6,   6.7, 5.1,   6.3, 10.1,  10.1,  19.1]),
        'beams': np.array([60, 56, 48, 43, 39, 35, 29, 25, 23, 21, 20, 19, 24, 20, 17]),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'planck',
        'use_smoothing': True,
    }
