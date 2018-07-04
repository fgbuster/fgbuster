#!/usr/bin/env python
import unittest
import numpy as np
from numpy.testing import assert_allclose as aac
from numpy.testing import assert_array_almost_equal as aaae
from fgbuster.pysm_helpers import get_instrument, get_sky
import fgbuster.component_model as cm
from fgbuster.separation_recipies import basic_comp_sep


class TestEnd2End(unittest.TestCase):

    def setUp(self):
        NSIDE = 32
        MODEL = 'c1d0s0f1'
        INSTRUMENT = 'litebird'
        X0_FACTOR = 1.01
        sky = get_sky(NSIDE, MODEL)
        self.instrument = get_instrument(NSIDE, INSTRUMENT)
        self.freq_maps = self.instrument.observe(sky, write_outputs=False)[0]

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
        aac(res_T.x, np.array(self.input), rtol=6)
        aaae(res_T.chi, 0, decimal=1) # FIXME otherwise doesn't pass in python3


    def test_basic_comp_sep_P(self):
        res_P = basic_comp_sep(self.components[:-1], self.instrument,
                               self.freq_maps[:, 1:, :])
        aac(res_P.x, np.array(self.input[:-1]), rtol=6)
        aaae(res_P.chi, 0, decimal=2)


if __name__ == '__main__':
    unittest.main()
