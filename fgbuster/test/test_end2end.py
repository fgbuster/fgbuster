#!/usr/bin/env python
import unittest
import sys
import os
import numpy as np
from numpy.testing import assert_allclose as aac
from numpy.testing import assert_array_almost_equal as aaae
from scipy.stats import kstest
import healpy as hp
from fgbuster.observation_helpers import get_instrument, get_sky
from fgbuster.algebra import _mtv
import fgbuster.component_model as cm
from fgbuster.separation_recipies import basic_comp_sep
import pysm
from pysm import convert_units

from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    # Borrowed from
    # https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class TestEnd2EndNoiselessPhysical(unittest.TestCase):

    def setUp(self):
        NSIDE = 16
        MODEL = 'c1d0s0f1'
        INSTRUMENT = 'litebird'
        X0_FACTOR = 0.99
        sky = pysm.Sky(get_sky(NSIDE, MODEL))
        self.instrument = pysm.Instrument(get_instrument(INSTRUMENT, NSIDE))
        with suppress_stdout():
            self.freq_maps, self.noise = self.instrument.observe(
                sky, write_outputs=False)

        self.components = [cm.CMB(), cm.Dust(200.), cm.Synchrotron(100.)]
        freefree = cm.PowerLaw(100.)
        freefree.defaults = [-2.14]  # Otherwise it is the same as Synchrotron
        self.components.append(freefree)
        self.input = []
        for component in self.components:
            self.input += component.defaults
            component.defaults = [d*X0_FACTOR for d in component.defaults]


    def test_basic_comp_sep_T(self):
        res_T = basic_comp_sep(self.components, self.instrument,
                               self.freq_maps[:, :1, :])
        aac(res_T.x, np.array(self.input), rtol=2e-5)
        aaae(res_T.chi, 0, decimal=2)


    def test_basic_comp_sep_P(self):
        res_P = basic_comp_sep(self.components[:-1], self.instrument,
                               self.freq_maps[:, 1:, :])
        aac(res_P.x, np.array(self.input[:-1]), rtol=1e-5)
        aaae(res_P.chi, 0, decimal=2)


class TestEnd2EndNoisy(unittest.TestCase):

    def test_dependence_on_nu0_RJ(self):
        NSIDE = 8
        MODEL = 'c1s0'
        INSTRUMENT = 'litebird'
        sky = pysm.Sky(get_sky(NSIDE, MODEL))
        instrument = pysm.Instrument(
            get_instrument(INSTRUMENT, NSIDE, units='uK_RJ'))
        components100 = [cm.CMB(units='K_RJ'),
                         cm.Synchrotron(100., units='K_RJ')]
        components10 = [cm.CMB(units='K_RJ'),
                        cm.Synchrotron(10., units='K_RJ')]

        with suppress_stdout():
            freq_maps, _ = instrument.observe(sky, write_outputs=False)

        res100 = basic_comp_sep(components100, instrument, freq_maps)
        res10 = basic_comp_sep(components10, instrument, freq_maps)
        aac(res100.Sigma, res10.Sigma)
        aac(res100.x, res10.x)
        aac(res100.s[0], res10.s[0])
        aac(res100.s[1], res10.s[1] * 10**res10.x[0])


    def test_dependence_on_nu0_CMB(self):
        NSIDE = 8
        MODEL = 'c1s0'
        INSTRUMENT = 'litebird'
        sky = pysm.Sky(get_sky(NSIDE, MODEL))
        instrument = pysm.Instrument(get_instrument(INSTRUMENT, NSIDE))
        components100 = [cm.CMB(), cm.Synchrotron(100.)]
        components10 = [cm.CMB(), cm.Synchrotron(10.)]

        with suppress_stdout():
            freq_maps, _ = instrument.observe(sky, write_outputs=False)

        res100 = basic_comp_sep(components100, instrument, freq_maps)
        res10 = basic_comp_sep(components10, instrument, freq_maps)
        aac(res100.Sigma, res10.Sigma)
        aac(res100.x, res10.x)
        aac(res100.s[0], res10.s[0])
        factor = convert_units('K_CMB', 'K_RJ', 10.)
        factor /= convert_units('K_CMB', 'K_RJ', 100.)
        aac(res100.s[1], res10.s[1] * 10**res10.x[0] * factor)


    def test_Sigma_synchrotron(self):
        NSIDE = 8
        MODEL = 's0'
        INSTRUMENT = 'litebird'
        SIGNAL_TO_NOISE = 20
        sky = pysm.Sky(get_sky(NSIDE, MODEL))
        instrument = pysm.Instrument(get_instrument(INSTRUMENT, NSIDE))
        components = [cm.Synchrotron(100.)]
        ref = []
        for component in components:
            ref += component.defaults

        with suppress_stdout():
            freq_maps, noise_maps = instrument.observe(sky, write_outputs=False)

        signal = freq_maps[:, 0, 0]
        noise = np.std(noise_maps[:, 0], axis=-1)
        maps = signal / np.dot(signal, noise) * SIGNAL_TO_NOISE
        maps = maps[:, np.newaxis] + noise_maps[:, 0] 
        res = basic_comp_sep(components, instrument,
                             maps, nside=hp.get_nside(maps))
        white = (res.x[0] - ref[0]) / res.Sigma[0, 0]**0.5
        _, p = kstest(white, 'norm')
        assert p > 0.01

    def test_Sigma_dust_one_parameter(self):
        NSIDE = 8
        MODEL = 'd0'
        INSTRUMENT = 'litebird'
        SIGNAL_TO_NOISE = 10
        sky = pysm.Sky(get_sky(NSIDE, MODEL))
        instrument = pysm.Instrument(get_instrument(INSTRUMENT, NSIDE))
        components = [cm.Dust(100., temp=20.)]
        ref = []
        for component in components:
            ref += component.defaults

        with suppress_stdout():
            freq_maps, noise_maps = instrument.observe(sky, write_outputs=False)

        signal = freq_maps[:, 0, 0]
        noise = noise_maps[:, 0]
        signal_ver =  signal / np.dot(signal, signal)**0.5
        noise_std = np.std([np.dot(n, signal_ver) for n in noise.T])
        maps = signal_ver * noise_std  * SIGNAL_TO_NOISE
        maps = maps[:, np.newaxis] + noise

        res = basic_comp_sep(components, instrument,
                             maps, nside=hp.get_nside(maps))
        white = (res.x[0] - ref[0]) / res.Sigma[0, 0]**0.5
        _, p = kstest(white, 'norm')
        assert p > 0.01

    def test_Sigma_dust_two_parameters(self):
        NSIDE = 8
        MODEL = 'd0'
        INSTRUMENT = 'litebird'
        SIGNAL_TO_NOISE = 10000
        sky = pysm.Sky(get_sky(NSIDE, MODEL))
        instrument = pysm.Instrument(get_instrument(INSTRUMENT, NSIDE))
        components = [cm.Dust(150.)]
        ref = []
        for component in components:
            ref += component.defaults
        ref = np.array(ref)

        with suppress_stdout():
            freq_maps, noise_maps = instrument.observe(sky, write_outputs=False)

        signal = freq_maps[:, 0, 0]  # Same signal for all the pixels
        noise = noise_maps[:, 0]
        signal_ver =  signal / np.dot(signal, signal)**0.5
        noise_std = np.std([np.dot(n, signal_ver) for n in noise.T])
        maps = signal_ver * noise_std  * SIGNAL_TO_NOISE
        maps = maps[:, np.newaxis] + noise

        res = basic_comp_sep(components, instrument, maps, nside=NSIDE)
        diff = (res.x.T - ref)
        postS = np.mean(diff[..., None] * diff[..., None, :], axis=0)
        S = res.Sigma.T[0]
        aac(postS, S, rtol=1./NSIDE)


    def test_Sigma_dust_sync_betas_temp(self):
        NSIDE = 8
        MODEL = 'd0s0'
        INSTRUMENT = 'litebird'
        SIGNAL_TO_NOISE = 10000
        UNITS = 'uK_CMB'
        sky = pysm.Sky(get_sky(NSIDE, MODEL))
        instrument = pysm.Instrument(
            get_instrument(INSTRUMENT, NSIDE, units=UNITS))
        components = [cm.Dust(150., temp=20., units=UNITS),
                      cm.Synchrotron(150., units=UNITS)]
        ref = []
        for component in components:
            ref += component.defaults
        ref = np.array(ref)

        with suppress_stdout():
            freq_maps, noise_maps = instrument.observe(sky, write_outputs=False)

        signal = freq_maps[:, 0, 0]  # Same signal for all the pixels
        noise = noise_maps[:, 0]
        signal_ver =  signal / np.dot(signal, signal)**0.5
        noise_std = np.std([np.dot(n, signal_ver) for n in noise.T])
        maps = signal_ver * noise_std  * SIGNAL_TO_NOISE
        maps = maps[:, np.newaxis] + noise

        res = basic_comp_sep(components, instrument, maps, nside=NSIDE)
        diff = (res.x.T - ref)
        postS = np.mean(diff[..., None] * diff[..., None, :], axis=0)
        S = res.Sigma.T[0]
        aac(postS, S, rtol=1./NSIDE)


if __name__ == '__main__':
    unittest.main()
