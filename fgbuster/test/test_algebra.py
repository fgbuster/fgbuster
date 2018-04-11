#!/usr/bin/env python
import numpy as np
from numpy.random import uniform
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_allclose as aac
import unittest
import fgbuster.component_model as cm
from ..mixingmatrix import MixingMatrix
from ..algebra import W, Wd, invAtNA, W_dB, W_dBdB, _mv, _mtm, _T

class TestAlgebraRandom(unittest.TestCase):

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


class TestAlgebraPhysical(unittest.TestCase):

    def setUp(self):
        self.NUM_DIF_DX = 1e-5
        np.random.seed(0)
        self.n_freq = 6
        self.nu = np.logspace(1, 2.5, self.n_freq)
        self.n_stokes = 3
        self.n_pixels = 10
        self.components = [cm.CMB(), cm.Dust(200.), cm.Synchrotron(100.)]
        self.mm = MixingMatrix(*(self.components))
        self.params = [1.54, 20, -3]
        self.A = self.mm.eval(self.nu, *(self.params))
        self.A_dB = self.mm.diff(self.nu, *(self.params))
        self.A_dBdB = self.mm.diff_diff(self.nu, *(self.params))
        self.invN = uniform(
            size=(self.n_pixels, self.n_stokes, self.n_freq, self.n_freq))
        self.invN += _T(self.invN)
        self.invN += 4*np.eye(self.n_freq)

    def test_W_dB_invN(self):
        W_dB_analytic = W_dB(self.A, self.A_dB, self.mm.comp_of_dB, self.invN)
        W_params = W(self.A, self.invN)
        for i in range(len(self.params)):
            diff_params = [p for p in self.params]
            diff_params[i] = self.NUM_DIF_DX + diff_params[i]
            diff_A = self.mm.eval(self.nu, *diff_params)
            diff_W = W(diff_A, self.invN)
            W_dB_numerical = (diff_W - W_params) / self.NUM_DIF_DX
            aac(W_dB_numerical, W_dB_analytic[i], rtol=self.NUM_DIF_DX*100)

    def test_W_dB(self):
        W_dB_analytic = W_dB(self.A, self.A_dB, self.mm.comp_of_dB)
        W_params = W(self.A)
        for i in range(len(self.params)):
            diff_params = [p for p in self.params]
            diff_params[i] = self.NUM_DIF_DX + diff_params[i]
            diff_A = self.mm.eval(self.nu, *diff_params)
            diff_W = W(diff_A)
            W_dB_numerical = (diff_W - W_params) / self.NUM_DIF_DX
            aac(W_dB_numerical, W_dB_analytic[i], rtol=self.NUM_DIF_DX*100)


if __name__ == '__main__':
    unittest.main()
