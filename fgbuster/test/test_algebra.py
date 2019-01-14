#!/usr/bin/env python
import unittest
import numpy as np
from numpy.random import uniform
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_allclose as aac
import fgbuster.component_model as cm
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.algebra import (W, Wd, invAtNA, W_dB, W_dBdB, _mv, _mtm, _mm, _T,
                              _mmm, D, comp_sep, multi_comp_sep, _mtmm, P,
                              P_dBdB)

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
        self.invN = uniform(
            size=(self.n_pixels, self.n_stokes, self.n_freq, self.n_freq))
        self.invN += _T(self.invN)
        self.invN += 10*np.eye(self.n_freq)
        self.invN *= 10

    def test_D(self):
        K = np.linalg.inv(_mtm(self.A, self.A))
        res = np.eye(self.n_freq) - _mmm(self.A, K, self.A.T)
        aaae(res, D(self.A))

    def test_D_invN(self):
        invN = self.invN[0,0]
        K = np.linalg.inv(_mtmm(self.A, invN, self.A))
        res = np.eye(self.n_freq) - _mmm(self.A, K, _mtm(self.A, invN))
        aaae(res, D(self.A, invN))

    def test_D_idempotence(self):
        invN = self.invN[0,0]
        res = D(self.A, invN)
        aaae(res, _mm(res, res))

    def test_P(self):
        K = np.linalg.inv(_mtm(self.A, self.A))
        res = _mmm(self.A, K, self.A.T)
        aaae(res, P(self.A))

    def test_P_invN(self):
        invN = self.invN[0,0]
        K = np.linalg.inv(_mtmm(self.A, invN, self.A))
        res = _mmm(self.A, K, _mtm(self.A, invN))
        aaae(res, P(self.A, invN))

    def test_P_idempotence(self):
        invN = self.invN[0,0]
        res = P(self.A, invN)
        aaae(res, _mm(res, res))

    def test_invAtNA(self):
        res = np.linalg.inv(_mtm(self.A, self.A))
        aaae(res, invAtNA(self.A))

    def test_invAtNA_invN(self):
        res = np.linalg.inv(_mtm(self.A, _mm(self.invN, self.A)))
        aaae(res, invAtNA(self.A, self.invN))

    def test_Wd_is_s(self):
        aaae(self.s, Wd(self.A, self.d))

    def test_W_on_d_is_s(self):
        aaae(self.s, _mv(W(self.A), self.d))

    def test_comp_sep_no_par(self):
        res = comp_sep(self.A, self.d, None, None, None)
        aaae(self.s, res.s)

    def test_multi_comp_sep_no_par(self):
        patch_ids = np.arange(self.d.shape[0]) // 2
        np.random.shuffle(patch_ids)
        res = multi_comp_sep(self.A, self.d, None, None, None, patch_ids)
        aaae(self.s, res.s)



class TestAlgebraPhysical(unittest.TestCase):

    def setUp(self):
        self.DX = 5e-4  # NOTE: this is a bit fine-tuned
        np.random.seed(0)
        self.n_freq = 6
        self.nu = np.logspace(1, 2.5, self.n_freq)
        self.n_stokes = 3
        self.n_pixels = 2
        self.components = [cm.CMB(), cm.Dust(200.), cm.Synchrotron(70.)]
        self.mm = MixingMatrix(*(self.components))
        self.params = [1.54, 20, -3]
        self.A = self.mm.eval(self.nu, *(self.params))
        self.A_dB = self.mm.diff(self.nu, *(self.params))
        self.A_dBdB = self.mm.diff_diff(self.nu, *(self.params))
        self.invN = uniform(
            size=(self.n_pixels, self.n_stokes, self.n_freq, self.n_freq))
        self.invN += _T(self.invN)
        self.invN += 10*np.eye(self.n_freq)
        self.invN *= 10

    def test_W_dB_invN(self):
        W_dB_analytic = W_dB(self.A, self.A_dB, self.mm.comp_of_dB, self.invN)
        W_params = W(self.A, self.invN)
        for i in range(len(self.params)):
            diff_params = [p for p in self.params]
            diff_params[i] = self.DX + diff_params[i]
            diff_A = self.mm.eval(self.nu, *diff_params)
            diff_W = W(diff_A, self.invN)
            W_dB_numerical = (diff_W - W_params) / self.DX
            aac(W_dB_numerical, W_dB_analytic[i], rtol=1e-3)

    def test_W_dB(self):
        W_dB_analytic = W_dB(self.A, self.A_dB, self.mm.comp_of_dB)
        W_params = W(self.A)
        for i in range(len(self.params)):
            diff_params = [p for p in self.params]
            diff_params[i] = self.DX + diff_params[i]
            diff_A = self.mm.eval(self.nu, *diff_params)
            diff_W = W(diff_A)
            W_dB_numerical = (diff_W - W_params) / self.DX
            aac(W_dB_numerical, W_dB_analytic[i], rtol=1e-3)

    def test_P_dBdB(self):
        P_dBdB_analytic = P_dBdB(
            self.A, self.A_dB, self.A_dBdB, self.mm.comp_of_dB)
        def get_P_displaced(i, j):
            def P_displaced(i_step, j_step):
                diff_params = [p for p in self.params]
                diff_params[i] = i_step * self.DX + diff_params[i]
                diff_params[j] = j_step * self.DX + diff_params[j]
                diff_A = self.mm.eval(self.nu, *diff_params)
                return P(diff_A)
            return P_displaced

        for i in range(len(self.params)):
            for j in range(len(self.params)):
                Pdx = get_P_displaced(i, j)
                if i == j:
                    P_dBdB_numerical = (
                        (-2*Pdx(0, 0) + Pdx(+1, 0) + Pdx(-1, 0)) / self.DX**2)
                else:
                    P_dBdB_numerical = (
                        (Pdx(1, 1) - Pdx(+1, -1) - Pdx(-1, 1) + Pdx(-1, -1))
                        / (4 * self.DX**2))
                aac(P_dBdB_numerical, P_dBdB_analytic[i][j], rtol=1.5e-1)

    def test_P_dBdB_invN(self):
        invN = self.invN[0,0]
        P_dBdB_analytic = P_dBdB(
            self.A, self.A_dB, self.A_dBdB, self.mm.comp_of_dB, invN)
        def get_P_displaced(i, j):
            def P_displaced(i_step, j_step):
                diff_params = [p for p in self.params]
                diff_params[i] = i_step * self.DX + diff_params[i]
                diff_params[j] = j_step * self.DX + diff_params[j]
                diff_A = self.mm.eval(self.nu, *diff_params)
                return P(diff_A, invN)
            return P_displaced

        for i in range(len(self.params)):
            for j in range(len(self.params)):
                Pdx = get_P_displaced(i, j)
                if i == j:
                    P_dBdB_numerical = (
                        (-2*Pdx(0, 0) + Pdx(+1, 0) + Pdx(-1, 0)) / self.DX**2)
                else:
                    P_dBdB_numerical = (
                        (Pdx(1, 1) - Pdx(+1, -1) - Pdx(-1, 1) + Pdx(-1, -1))
                        / (4 * self.DX**2))
                aac(P_dBdB_numerical, P_dBdB_analytic[i][j], rtol=3.0e-1)

    def test_W_dBdB(self):
        W_dBdB_analytic = W_dBdB(
            self.A, self.A_dB, self.A_dBdB, self.mm.comp_of_dB)
        def get_W_displaced(i, j):
            def W_displaced(i_step, j_step):
                diff_params = [p for p in self.params]
                diff_params[i] = i_step * self.DX + diff_params[i]
                diff_params[j] = j_step * self.DX + diff_params[j]
                diff_A = self.mm.eval(self.nu, *diff_params)
                return W(diff_A)
            return W_displaced

        for i in range(len(self.params)):
            for j in range(len(self.params)):
                Wdx = get_W_displaced(i, j)
                if i == j:
                    W_dBdB_numerical = (
                        (-2*Wdx(0, 0) + Wdx(+1, 0) + Wdx(-1, 0)) / self.DX**2)
                else:
                    W_dBdB_numerical = (
                        (Wdx(1, 1) - Wdx(+1, -1) - Wdx(-1, 1) + Wdx(-1, -1))
                        / (4 * self.DX**2))
                aac(W_dBdB_numerical, W_dBdB_analytic[i][j], rtol=1e-1)

    def test_W_dBdB_invN(self):
        W_dBdB_analytic = W_dBdB(
            self.A, self.A_dB, self.A_dBdB, self.mm.comp_of_dB, self.invN)
        def get_W_displaced(i, j):
            def W_displaced(i_step, j_step):
                diff_params = [p for p in self.params]
                diff_params[i] = i_step * self.DX + diff_params[i]
                diff_params[j] = j_step * self.DX + diff_params[j]
                diff_A = self.mm.eval(self.nu, *diff_params)
                return W(diff_A, self.invN)
            return W_displaced

        for i in range(len(self.params)):
            for j in range(len(self.params)):
                Wdx = get_W_displaced(i, j)
                if i == j:
                    W_dBdB_numerical = (
                        (-2*Wdx(0, 0) + Wdx(+1, 0) + Wdx(-1, 0)) / self.DX**2)
                else:
                    W_dBdB_numerical = (
                        (Wdx(1, 1) - Wdx(+1, -1) - Wdx(-1, 1) + Wdx(-1, -1))
                        / (4 * self.DX**2))
                aac(W_dBdB_numerical, W_dBdB_analytic[i][j], rtol=2.5e-1)


if __name__ == '__main__':
    unittest.main()
