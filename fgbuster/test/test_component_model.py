#!/usr/bin/env python
import unittest
from itertools import product
from parameterized import parameterized
import scipy
import numpy as np
from fgbuster.component_model import AnalyticComponent, Dust
from fgbuster.observation_helpers import get_sky, get_instrument, _jysr2rj
import pysm3
import pysm3.units as u

class TestModifiedBlackBody(unittest.TestCase):

    def setUp(self):
        self.freqs = np.array([50, 100, 300])
        self.temp = 19.6
        self.beta_d = 1.7
        self.dust_t_b = Dust(150.)
        self.dust_b = Dust(150., temp=self.temp)
        self.dust_t = Dust(150., beta_d=self.beta_d)
        self.dust = Dust(150., temp=self.temp, beta_d=self.beta_d)
    
    def test_init_and_evaluation_parameters(self):
        x = self.dust_t_b.eval(self.freqs, self.beta_d, self.temp)
        np.testing.assert_array_almost_equal(
            x, self.dust_t.eval(self.freqs, self.temp))
        np.testing.assert_array_almost_equal(
            x, self.dust_b.eval(self.freqs, self.beta_d))
        np.testing.assert_array_almost_equal(
            x, self.dust.eval(self.freqs))


class TestAnalyticComponent(unittest.TestCase):
    
    funcs = ['eval', 'diff']
    vals = ['float', 'scal', 'vec', 'vecbcast']
    bands = ['centers', 'bandpass']
    tags = ['__'.join(args) for args in product(funcs, vals, vals, bands)]

    def setUp(self):
        self.analitic_expr = 'nu * param0 + nu**param1 + hundred'
        self.comp = AnalyticComponent(self.analitic_expr, hundred=100)

    def hard_eval(self, nu, param0, param1):
        param0 = self._add_dim_if_ndarray(param0)
        param1 = self._add_dim_if_ndarray(param1)
        nu = np.array(nu)
        if nu.ndim == 1: # No bandpass
            return nu * param0 + nu**param1 + 100.
        elif nu.ndim == 3: # Bandpass and trasmission
            nu, weight = np.swapaxes(nu, 0, 1)
            nu_shape = nu.shape
            nu = nu.flatten()
            res = nu * param0 + nu**param1 + 100.
            res *= weight.flatten()
            return np.trapz(res.reshape(res.shape[:-1] + nu_shape),
                            nu.reshape(nu_shape) * 1e9, -1)
        raise

    def hard_diff(self, nu, param0, param1):
        param0 = self._add_dim_if_ndarray(param0)
        param1 = self._add_dim_if_ndarray(param1)
        nu = np.array(nu)
        if nu.ndim == 1: # No bandpass
            return [nu, nu**param1 * np.log(nu)]
        elif nu.ndim == 3: # Bandpass and trasmission
            nu, weight = np.swapaxes(nu, 0, 1)
            nu_shape = nu.shape
            nu = nu.flatten()
            res = nu**param1 * np.log(nu)
            res *= weight.flatten()
            nu = nu.reshape(nu_shape)
            res = np.trapz(res.reshape(res.shape[:-1] + nu_shape), nu * 1e9, -1)
            return [np.trapz(nu*weight, nu * 1e9, -1), res]
        raise

    def _add_dim_if_ndarray(self, param):
        if isinstance(param, np.ndarray):
            return param[..., np.newaxis]
        return param

    def _get_nu(self, tag):
        if tag == 'centers':
            return np.arange(1, 4) * 10
        elif tag == 'bandpass':
            bandpass = np.arange(1, 4) * 10
            bandpass = bandpass[:, np.newaxis] * np.linspace(0.9, 1.1, 11)
            weight = np.random.uniform(0.5, 1.5, bandpass.size)
            weight = weight.reshape(bandpass.shape)
            return tuple(np.swapaxes(np.stack((bandpass, weight)), 0, 1))
        raise ValueError(tag)

    def _get_param0(self, tag):
        if tag == 'float':
            return 2.
        elif tag == 'scal':
            return np.array([2.])
        elif tag == 'vec':
            return np.arange(2, 7)
        elif tag == 'vecbcast':
            return np.arange(2, 7)[:, np.newaxis]
        raise ValueError(tag)

    def _get_param1(self, tag):
        if tag == 'float':
            return 1.5
        elif tag == 'scal':
            return np.array([1.5])
        elif tag == 'vec':
            return np.linspace(1.5, 2., 5)
        elif tag == 'vecbcast':
            return np.linspace(1.5, 2., 5)[:, np.newaxis, np.newaxis]
        raise ValueError(tag)

    @parameterized.expand(tags)
    def test(self, tag):
        func, val0, val1, nu_type = tag.split('__')
        param0 = self._get_param0(val0)
        param1 = self._get_param1(val1)
        nu = self._get_nu(nu_type)

        res = getattr(self.comp, func)(nu, param0, param1)

        ref = getattr(self, 'hard_'+func)(nu, param0, param1)

        if not isinstance(res, list):
            res = [res]
            ref = [ref]

        for args in zip(res, ref):
            np.testing.assert_allclose(*args)

    def test_bandpass_integration_against_pysm(self):
        NSIDE = 2
        N_SAMPLE_BAND = 10
        sky = get_sky(NSIDE, 'd1')
        freqs = np.linspace(80, 120, N_SAMPLE_BAND)
        weights = np.ones(N_SAMPLE_BAND)
        weights /= np.trapz(weights, freqs*1e9)
        pysm_map = sky.get_emission(freqs * u.GHz, weights)[1].value  # Select Q

        weights = weights / _jysr2rj(freqs)
        weights /= np.trapz(weights, freqs * 1e9)
        dust = sky.components[0]
        fgb_map = Dust(dust.freq_ref_P.value, units='uK_RJ').eval(
            [(freqs, weights)], dust.mbb_index.value, dust.mbb_temperature.value)
        fgb_map = fgb_map[..., 0] * dust.Q_ref.value
        np.testing.assert_allclose(pysm_map, fgb_map, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
