#!/usr/bin/env python
import numpy as np
from numpy.random import uniform
from numpy.testing import assert_array_almost_equal as aaae
import unittest
from ..component_model import Component, Dust
from ..algebra import W, Wd, invAtNA, _mv, _mtm

class TestAlgebra(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.n_freq = 6
        self.n_comp = 2
        self.n_stokes = 3
        self.n_pixels = 10
        self.A = uniform(size=(self.n_freq, self.n_comp))
        self.s = uniform(size=(self.n_pixels, self.n_stokes, self.n_comp))
        self.d = _mv(self.A, self.s)
    
    def test_invAtNA(self):
        res = np.linalg.inv(_mtm(self.A, self.A))
        aaae(res, invAtNA(self.A))

    def test_Wd_is_s(self):
        aaae(self.s, Wd(self.A, self.d))

    def test_W_on_d_is_s(self):
        aaae(self.s, _mv(W(self.A), self.d))


if __name__ == '__main__':
    unittest.main()
