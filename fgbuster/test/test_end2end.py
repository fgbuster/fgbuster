#!/usr/bin/env python
import unittest
import numpy as np
from numpy.testing import assert_allclose as aac
from scipy.stats import kstest
import healpy as hp
from fgbuster.observation_helpers import (get_instrument, get_noise_realization,
                                          get_sky, get_observation,
                                          _rj2cmb, _cmb2rj)
import fgbuster.component_model as cm
from fgbuster.separation_recipes import basic_comp_sep


class TestEnd2EndNoiselessPhysical(unittest.TestCase):

    def setUp(self):
        NSIDE = 16
        MODEL = 'c1d0s0f1'
        INSTRUMENT = 'LiteBIRD'
        X0_FACTOR = 0.99
        sky = get_sky(NSIDE, MODEL)
        self.instrument = get_instrument(INSTRUMENT)
        self.freq_maps = get_observation(self.instrument, sky)

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
                               self.freq_maps[:, :1, :], tol=1e-12)
        aac(res_T.x, np.array(self.input), rtol=1e-4)
        aac(res_T.chi, 0, atol=0.05)

    def test_basic_comp_sep_P(self):
        res_P = basic_comp_sep(self.components[:-1], self.instrument,
                               self.freq_maps[:, 1:, :])
        aac(res_P.x, np.array(self.input[:-1]), rtol=1e-5)
        aac(res_P.chi, 0, atol=0.01)


class TestEnd2EndNoisy(unittest.TestCase):

    def test_dependence_on_nu0_RJ(self):
        NSIDE = 8
        MODEL = 'c1s0'
        INSTRUMENT = 'LiteBIRD'
        sky = get_sky(NSIDE, MODEL)
        instrument = get_instrument(INSTRUMENT)
        components100 = [cm.CMB(units='K_RJ'),
                         cm.Synchrotron(100., units='K_RJ')]
        components10 = [cm.CMB(units='K_RJ'),
                        cm.Synchrotron(10., units='K_RJ')]

        freq_maps = get_observation(instrument, sky, unit='K_RJ')

        res100 = basic_comp_sep(components100, instrument, freq_maps)
        res10 = basic_comp_sep(components10, instrument, freq_maps)
        aac(res100.Sigma, res10.Sigma)
        aac(res100.x, res10.x)
        aac(res100.s[0], res10.s[0])
        aac(res100.s[1], res10.s[1] * 10**res10.x[0])


    def test_dependence_on_nu0_CMB(self):
        NSIDE = 4
        MODEL = 'c1s0'
        INSTRUMENT = 'LiteBIRD'
        sky = get_sky(NSIDE, MODEL)
        instrument = get_instrument(INSTRUMENT)
        components100 = [cm.CMB(), cm.Synchrotron(100.)]
        components10 = [cm.CMB(), cm.Synchrotron(10.)]

        freq_maps = get_observation(instrument, sky)

        res10 = basic_comp_sep(components10, instrument, freq_maps)
        res100 = basic_comp_sep(components100, instrument, freq_maps)
        aac(res100.Sigma, res10.Sigma)
        aac(res100.x, res10.x)
        aac(res100.s[0], res10.s[0], atol=1e-7)
        factor = _cmb2rj(10.) * _rj2cmb(100.)
        aac(res100.s[1], res10.s[1] * 10**res10.x[0] * factor)


    def test_Sigma_synchrotron(self):
        NSIDE = 8
        MODEL = 's0'
        INSTRUMENT = 'LiteBIRD'
        SIGNAL_TO_NOISE = 20
        sky = get_sky(NSIDE, MODEL)
        instrument = get_instrument(INSTRUMENT)
        components = [cm.Synchrotron(100.)]
        ref = []
        for component in components:
            ref += component.defaults

        freq_maps = get_observation(instrument, sky)
        noise_maps = get_noise_realization(NSIDE, instrument)

        signal = freq_maps[:, 0, 0]
        noise = np.std(noise_maps[:, 0], axis=-1)
        maps = signal / np.dot(signal, noise) * SIGNAL_TO_NOISE
        maps = maps[:, np.newaxis] + noise_maps[:, 0]
        if not hasattr(instrument, 'depth_i'):
            instrument['depth_i'] = instrument.depth_p / np.sqrt(2)
        res = basic_comp_sep(components, instrument,
                             maps, nside=hp.get_nside(maps))
        white = (res.x[0] - ref[0]) / res.Sigma[0, 0]**0.5
        _, p = kstest(white, 'norm')
        assert p > 0.01, f'KS probability is {p}'

    def test_Sigma_dust_one_parameter(self):
        NSIDE = 8
        MODEL = 'd0'
        INSTRUMENT = 'LiteBIRD'
        SIGNAL_TO_NOISE = 10
        sky = get_sky(NSIDE, MODEL)
        instrument = get_instrument(INSTRUMENT)
        components = [cm.Dust(100., temp=20.)]
        ref = []
        for component in components:
            ref += component.defaults

        freq_maps = get_observation(instrument, sky)
        noise_maps = get_noise_realization(NSIDE, instrument)

        signal = freq_maps[:, 0, 0]
        noise = noise_maps[:, 0]
        signal_ver = signal / np.dot(signal, signal)**0.5
        noise_std = np.std([np.dot(n, signal_ver) for n in noise.T])
        maps = signal_ver * noise_std  * SIGNAL_TO_NOISE
        maps = maps[:, np.newaxis] + noise
        if not hasattr(instrument, 'depth_i'):
            instrument['depth_i'] = instrument.depth_p / np.sqrt(2)

        res = basic_comp_sep(components, instrument,
                             maps, nside=hp.get_nside(maps))
        white = (res.x[0] - ref[0]) / res.Sigma[0, 0]**0.5
        _, p = kstest(white, 'norm')
        assert p > 0.01

    def test_Sigma_dust_two_parameters(self):
        NSIDE = 8
        MODEL = 'd0'
        INSTRUMENT = 'LiteBIRD'
        SIGNAL_TO_NOISE = 10000
        sky = get_sky(NSIDE, MODEL)
        instrument = get_instrument(INSTRUMENT)
        components = [cm.Dust(150.)]
        ref = []
        for component in components:
            ref += component.defaults
        ref = np.array(ref)

        freq_maps = get_observation(instrument, sky)
        noise_maps = get_noise_realization(NSIDE, instrument)

        signal = freq_maps[:, 0, 0]  # Same signal for all the pixels
        noise = noise_maps[:, 0]
        signal_ver = signal / np.dot(signal, signal)**0.5
        noise_std = np.std([np.dot(n, signal_ver) for n in noise.T])
        maps = signal_ver * noise_std  * SIGNAL_TO_NOISE
        maps = maps[:, np.newaxis] + noise
        if not hasattr(instrument, 'depth_i'):
            instrument['depth_i'] = instrument.depth_p / np.sqrt(2)

        res = basic_comp_sep(components, instrument, maps, nside=NSIDE)
        diff = (res.x.T - ref)
        postS = np.mean(diff[..., None] * diff[..., None, :], axis=0)
        S = res.Sigma.T[0]
        aac(postS, S, rtol=1./NSIDE)


    def test_Sigma_dust_sync_betas_temp(self):
        NSIDE = 8
        MODEL = 'd0s0'
        INSTRUMENT = 'LiteBIRD'
        SIGNAL_TO_NOISE = 10000
        UNITS = 'uK_CMB'
        sky = get_sky(NSIDE, MODEL)
        instrument = get_instrument(INSTRUMENT)
        components = [cm.Dust(150., temp=20., units=UNITS),
                      cm.Synchrotron(150., units=UNITS)]
        ref = []
        for component in components:
            ref += component.defaults
        ref = np.array(ref)

        freq_maps = get_observation(instrument, sky, unit=UNITS)
        noise_maps = get_noise_realization(NSIDE, instrument, unit=UNITS)

        signal = freq_maps[:, 0, 0]  # Same signal for all the pixels
        noise = noise_maps[:, 0]
        signal_ver =  signal / np.dot(signal, signal)**0.5
        noise_std = np.std([np.dot(n, signal_ver) for n in noise.T])
        maps = signal_ver * noise_std  * SIGNAL_TO_NOISE
        maps = maps[:, np.newaxis] + noise
        if not hasattr(instrument, 'depth_i'):
            instrument['depth_i'] = instrument.depth_p / np.sqrt(2)

        res = basic_comp_sep(components, instrument, maps, nside=NSIDE)
        diff = (res.x.T - ref)
        postS = np.mean(diff[..., None] * diff[..., None, :], axis=0)
        S = res.Sigma.T[0]
        aac(postS, S, rtol=1./NSIDE)

    def test_chi(self):
        NSIDE = 64
        MODEL = 'd0s0'
        INSTRUMENT = 'LiteBIRD'
        UNITS = 'uK_CMB'
        sky = get_sky(NSIDE, MODEL)
        instrument = get_instrument(INSTRUMENT)
        components = [cm.Dust(150., beta_d=1.54, temp=20., units=UNITS),
                      cm.Synchrotron(150., -3., units=UNITS)]

        
        freq_maps = get_observation(instrument, sky, unit=UNITS)
        noise_maps = get_noise_realization(NSIDE, instrument, unit=UNITS)

        res = basic_comp_sep(components, instrument, freq_maps)
        snr = res.s.copy()
        snr[0] /= res.invAtNA[:, 0, 0][:, None]
        snr[1] /= res.invAtNA[:, 1, 1][:, None]
        snr = snr.max(axis=(0, 1))
        mask_snr = snr < 1e6
        chi = res.chi[..., mask_snr]
        dof = freq_maps.size - res.s.size
        dof *= mask_snr.mean()
        aac(chi, 0, atol=1e-3)
        res = basic_comp_sep(components, instrument, freq_maps + noise_maps)
        chi = res.chi[..., mask_snr]
        aac((chi**2).sum(), dof, atol=3*dof**0.5)


if __name__ == '__main__':
    unittest.main()
