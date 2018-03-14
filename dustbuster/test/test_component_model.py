#!/usr/bin/env python
import numpy as np
import unittest
from ..component_model import Component, Dust

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


class TestComponent(unittest.TestCase):

    def test_evaluate_all_vec(self):
        comp = Component('nu * param_0 + param_1 + hundred', hundred=100)
        nu = np.arange(1,3) * 10
        mult = np.arange(1,4)
        add = np.arange(3)
        res = nu * mult[..., np.newaxis] + add[..., np.newaxis] + 100
        np.testing.assert_array_almost_equal(comp.eval(nu, mult, add), res)

    def test_evaluate_vec_scalar(self):
        comp = Component('nu * param_0 + param_1 + hundred', hundred=100)
        nu = np.arange(1,3) * 10
        mult = 1
        add = np.arange(3)
        res = nu * mult + add[..., np.newaxis] + 100
        np.testing.assert_array_almost_equal(comp.eval(nu, mult, add), res)

    def test_evaluate_all_scalar(self):
        comp = Component('nu * param_0 + param_1 + hundred', hundred=100)
        nu = np.arange(1,3) * 10
        mult = 1
        add = 2
        res = nu * mult + add + 100
        np.testing.assert_array_almost_equal(comp.eval(nu, mult, add), res)


if __name__ == '__main__':
    unittest.main()
