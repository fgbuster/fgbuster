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

""" 
Handy access to instrument configuration, map generation
and other pysm3 functionalities
"""
import types
import numpy as np
import pandas as pd
import healpy as hp
import pysm3
import pysm3.units as u
from cmbdb import cmbdb


__all__ = [
    'get_sky',
    'get_instrument',
    'get_observation',
    'get_noise_realization',
]

INSTRUMENT_STD_ATTR = 'frequency depth_i depth_p fwhm'.split()
_NL = '\n'

def get_sky(nside, tag='c1d0s0'):
    """ Get a pre-defined PySM sky

    Parameters
    ----------
    nside: int
        healpix nside of the sky templates
    tag: string
        See the `pysm documentation
        <https://pysm3.readthedocs.io/en/latest/models.html#models>`_
        for a complete list of available options.
        Default is 'c1d0s0', i.e. cmb (c1), dust with constant temperature and
        spectral index (d0), and synchrotron with constant spectral index (s0).

    Returns
    -------
    sky: pysm3.Sky
        See the `pysm documentation
        <https://pysm3.readthedocs.io/en/latest/api/pysm.Sky.html#pysm.Sky>`_
    """
    preset_strings = [tag[i:i+2] for i in range(0, len(tag), 2)]
    return pysm3.Sky(nside, preset_strings=preset_strings)


def get_instrument(tag=''):
    """ Get a pre-defined instrumental configuration

    Parameters
    ----------
    tag: string
        name of the pre-defined experimental configurations.
        It can contain the name of multiple experiments separated by a space.
        Call the function with a random input to get the available instruments.

    Returns
    -------
    instr: pandas.DataFrame
        It contains the experimetnal configuration of the desired instrument(s).
    """
    df = cmbdb.loc[cmbdb['experiment'].isin(tag.split())]
    if df.empty:
        if tag == 'test':
            df = pd.DataFrame()
            df['frequency'] = np.arange(10., 300, 30.)
            df['depth_p'] = (np.linspace(20, 40, 10) - 30)**2
            df['depth_i'] = (np.linspace(20, 40, 10) - 30)**2
        else:
            from importlib.util import find_spec
            exp_file = find_spec('cmbdb').submodule_search_locations[0]
            exp_file += '/experiments.yaml'
            github = 'https://github.com/dpole/cmbdb'
            raise ValueError(
                (f"Instrument(s) {tag} not available." if tag else "") +
                f"Choose between: {' '.join(cmbdb.experiment.unique())}{_NL}"
                f"Add your instrument to your local copy of cmbdb: {exp_file}\n"
                f"Beware, you might lose changes when you update: "
                f"push your new configuration to {github}")
    return df.dropna(1, 'all')


def get_observation(instrument='', sky=None,
                    noise=False, nside=None, unit='uK_CMB'):
    """ Get a pre-defined instrumental configuration

    Parameters
    ----------
    instrument:
        It can be either a `str` (see :func:`get_instrument`) or an
        object that provides the following as a key or an attribute.

        - **frequency** (required)
        - **depth_p** (required if ``noise=True``)
        - **depth_i** (required if ``noise=True``)

        They can be anything that is convertible to a float numpy array.
        If only one of ``depth_p`` or ``depth_i`` is provided, the other is
        inferred assuming that the former is sqrt(2) higher than the latter.
    sky: str of pysm3.Sky
        Sky to observe. It can be a `pysm3.Sky` or a tag to create one.
    noise: bool
        If true, add Gaussian, uncorrelated, isotropic noise.
    nside: int
        Desired output healpix nside. It is optional if `sky` is a `pysm3.Sky`,
        and required if it is a `str` or ``None``.
    unit: str
        Unit of the output. Only K_CMB and K_RJ (and multiples) are supported.

    Returns
    -------
    observation: array
        Shape is ``(n_freq, 3, n_pix)``
    """
    if isinstance(instrument, str):
        instrument = get_instrument(instrument)
    else:
        instrument = standardize_instrument(instrument)
    if nside is None:
        nside = sky.nside
    elif not isinstance(sky, str):
        try:
            assert nside == sky.nside, (
                "Mismatch between the value of the nside of the pysm3.Sky "
                "argument and the one passed in the nside argument.")
        except AttributeError:
            raise ValueError("Either provide a pysm3.Sky as sky argument "
                             " or specify the nside argument.")

    if noise:
        res = get_noise_realization(nside, instrument, unit)
    else:
        res = np.zeros((len(instrument.frequency), 3, hp.nside2npix(nside)))

    if sky is None or sky == '':
        return res

    if isinstance(sky, str):
        sky = get_sky(nside, sky)

    for res_freq, freq in zip(res, instrument.frequency):
        emission = sky.get_emission(freq * u.GHz).to(
            getattr(u, unit),
            equivalencies=u.cmb_equivalencies(freq * u.GHz))
        res_freq += emission.value

    return res


def get_noise_realization(nside, instrument, unit='uK_CMB'):
    """ Generate noise maps for the instrument

    Parameters
    ----------
    nside: int
        Desired output healpix nside.
    instrument:
        Object that provides the following as a key or an attribute.

        - **frequency** (required)
        - **depth_p** (required if ``noise=True``)
        - **depth_i** (required if ``noise=True``)

        They can be anything that is convertible to a float numpy array.
        If only one of ``depth_p`` or ``depth_i`` is provided, the other is
        inferred assuming that the former is sqrt(2) higher than the latter.
    unit: str
        Unit of the output. Only K_CMB and K_RJ (and multiples) are supported.
    sky: str of pysm3.Sky
        Sky to observe. It can be a `pysm3.Sky` or a tag to create one.
    noise: bool
        If true, add Gaussian, uncorrelated, isotropic noise.

    Returns
    -------
    observation: array
        Shape is ``(n_freq, 3, n_pix)``.
    """
    instrument = standardize_instrument(instrument)

    n_freq = len(instrument.frequency)
    n_pix = hp.nside2npix(nside)
    res = np.random.normal(size=(n_pix, 3, n_freq))
    depth = np.stack(
        (instrument.depth_i, instrument.depth_p, instrument.depth_p))
    depth *= u.arcmin * u.uK_CMB
    depth = depth.to(
        getattr(u, unit) * u.arcmin,
        equivalencies=u.cmb_equivalencies(instrument.frequency * u.GHz))
    res *= depth.value / hp.nside2resol(nside, True)
    return res.T


def standardize_instrument(instrument):
    f"""Handle different input instruments

    Parameters
    ----------
    instrument:
        Anything that has

    {_NL.join(['    * '+attr for attr in INSTRUMENT_STD_ATTR]) }

        as keys or attributes, including `pandas.DataFrame`.

    Returns
    -------
    std_instr: SimpleNamespace
        It contains the properties above as attributes. They are converted to a
        float array.
    """
    std_instr = types.SimpleNamespace()
    for attr in INSTRUMENT_STD_ATTR:
        try:
            try:
                value = np.array(getattr(instrument, attr), dtype=np.float64)
            except AttributeError:
                value = np.array(instrument[attr], dtype=np.float64)
            setattr(std_instr, attr, value.copy())
        except (TypeError, KeyError):  # Not subscriptable or missing key
            pass
        if attr == 'frequency' and std_instr.frequency.ndim == 3:
            std_instr.frequency = [tuple(x) for x in std_instr.frequency]

    return std_instr


def _rj2cmb(freqs):
    return (np.ones_like(freqs) * u.K_RJ).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _cmb2rj(freqs):
    return (np.ones_like(freqs) * u.K_CMB).to(
        u.K_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value

def _rj2jysr(freqs):
    return (np.ones_like(freqs) * u.K_RJ).to(
        u.Jy / u.sr, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _jysr2rj(freqs):
    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _cmb2jysr(freqs):
    return (np.ones_like(freqs) * u.K_CMB).to(
        u.Jy / u.sr, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _jysr2cmb(freqs):
    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value
