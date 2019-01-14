#!/usr/bin/env python
import unittest
from itertools import product
from parameterized import parameterized
import numpy as np
from fgbuster.component_model import AnalyticComponent, Dust

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
    tags = ['__'.join(args) for args in product(funcs, vals, vals)]

    def setUp(self):
        self.analitic_expr = 'nu * param0 + nu**param1 + hundred'
        self.comp = AnalyticComponent(self.analitic_expr, hundred=100)
        self.nu = np.arange(1,4) * 10

    def hard_eval(self, nu, param0, param1):
        param0 = self._add_dim_if_ndarray(param0)
        param1 = self._add_dim_if_ndarray(param1)
        return nu * param0 + nu**param1 + 100.

    def hard_diff(self, nu, param0, param1):
        param0 = self._add_dim_if_ndarray(param0)
        param1 = self._add_dim_if_ndarray(param1)
        return [nu, nu**param1 * np.log(nu)]

    def _add_dim_if_ndarray(self, param):
        if isinstance(param, np.ndarray):
            return param[..., np.newaxis]
        return param

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
        func, val0, val1 = tag.split('__')
        param0 = self._get_param0(val0)
        param1 = self._get_param1(val1)

        res = getattr(self.comp, func)(self.nu, param0, param1)

        ref = getattr(self, 'hard_'+func)(self.nu, param0, param1)

        if not isinstance(res, list):
            res = [res]
            ref = [ref]

        for args in zip(res, ref):
            np.testing.assert_allclose(*args)


if __name__ == '__main__':
    unittest.main()
