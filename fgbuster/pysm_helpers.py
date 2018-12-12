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


def get_instrument(nside, tag, units='uK_CMB'):

    """ Get pre-defined PySM Instrument

    Parameters
    ----------
    tag: string
        name of the pre-defined experimental configurations. See the source or
        set tag to something random to have a list of the available
        configurations. it can contain the name of multiple experiments
        separated by a space.

    Returns
    -------
    sky: PySM.Instrument
    """
    module = sys.modules[__name__]
    instruments = []
    for t in tag.split():
        try:
            instrument = getattr(module, '_dict_instrument_'+t)(nside, units)
        except AttributeError:
            raise ValueError('Instrument %s not available. Chose between: %s.'
                             % (t, ', '.join(_get_available_instruments())))
        else:
            instruments.append(instrument)

    for key in instruments[0]:
        if isinstance(instruments[0][key], np.ndarray):
            instruments[0][key] = np.concatenate([i[key] for i in instruments])

    instruments[0]['prefix'] = '__'.join(tag.split())

    return pysm.Instrument(instruments[0])


def _get_available_instruments():
    prefix = '_dict_instrument_'
    module = sys.modules[__name__]
    has_prefix = lambda x: x.find(prefix) != -1
    return [fun.replace(prefix, '') for fun in dir(module) if has_prefix(fun)]


def _dict_instrument_test(nside, units='uK_CMB'):
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
        'use_smoothing': False,
    }


def _dict_instrument_litebird(nside, units='uK_CMB'):
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


def _dict_instrument_cmbs4(nside, units='uK_CMB'):
    # specifications taken from https://cmb-s4.org/wiki/index.php/Survey_Performance_Expectations
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


def _dict_instrument_quijote_mfi(nside, units='uK_CMB'):
    return {
        'frequencies': np.array([11.0, 13.0, 17.0, 19.0]),
        'sens_I': np.array([2100, 2100, 2100, 2100]) / 1.41,
        'sens_P': np.array([2100, 2100, 2100, 2100]),
        'beams': np.array([55.2, 55.2, 36.0, 36.0]),
        'nside': nside,
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_units': units,
        'output_directory': '/dev/null',
        'output_prefix': 'quijote_mfi',
        'use_smoothing': False,
    }

def _dict_instrument_quijote_supermfi(nside, units='uK_CMB'):
    mfi = _dict_instrument_quijote_mfi(nside, units)
    mfi['sens_I'] /= 100
    mfi['sens_P'] /= 100
    mfi['output_prefix'] = 'quijote_supermfi'
    return mfi
