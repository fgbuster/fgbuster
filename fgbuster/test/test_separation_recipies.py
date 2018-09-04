#!/usr/bin/env python
import unittest
import numpy as np
from numpy.testing import assert_allclose as aac
from numpy.testing import assert_array_almost_equal as aaae
from scipy.stats import kstest
import healpy as hp
from fgbuster.pysm_helpers import get_instrument, get_sky
from fgbuster.algebra import _mtv
import fgbuster.component_model as cm
from fgbuster.separation_recipies import basic_comp_sep


class TestBasicCompSep(unittest.TestCase):

    def test_T_s_2_no_param(self):
        NSIDE = 2
        NSIDE_PARAM = 0
        INSTRUMENT = 'test'
        power = 1.5

        np.random.seed(0)
        instrument = get_instrument(NSIDE, INSTRUMENT)
        components = [cm.Component('(nu - nu0)**power', nu0=30, power=power)]
        freq_maps = components[0].eval(instrument.Frequencies)
        s = np.linspace(1, 2, hp.nside2npix(NSIDE))
        np.random.shuffle(s)
        freq_maps = freq_maps[:, np.newaxis] * s

        res = basic_comp_sep(components, instrument,
                             freq_maps, nside=NSIDE_PARAM)
        aac(res.s.flatten(), s, rtol=1e-5)
        aaae(res.chi, 0, decimal=2)

    def test_s_3_no_param(self):
        NSIDE = 2
        NSIDE_PARAM = 0
        INSTRUMENT = 'test'
        power = 1.5

        np.random.seed(0)
        instrument = get_instrument(NSIDE, INSTRUMENT)
        components = [[cm.Component('(nu - nu0)**power', nu0=30, power=power)],
                      [cm.Component('(nu - nu0)**power', nu0=30, power=power)]]
        freq_maps = components[0][0].eval(instrument.Frequencies)
        s = np.linspace(1, 2, hp.nside2npix(NSIDE)*3)
        np.random.shuffle(s)
        freq_maps = freq_maps[:, np.newaxis] * s
        freq_maps = freq_maps.reshape(freq_maps.shape[0], 3, -1)

        res = basic_comp_sep(components, instrument,
                             freq_maps, nside=NSIDE_PARAM)
        aac(res.s.flatten(), s, rtol=1e-5)
        aaae(res.chi, 0, decimal=2)

    def test_T_s_2_param_1(self):
        NSIDE = 2
        NSIDE_PARAM = 1
        INSTRUMENT = 'test'

        np.random.seed(0)
        instrument = get_instrument(NSIDE, INSTRUMENT)
        components = [cm.Component('(nu - nu0)**power', nu0=30)]
        components[0].defaults = [1.]
        powers = np.linspace(1, 2, hp.nside2npix(NSIDE_PARAM))
        np.random.shuffle(powers)
        freq_maps = components[0].eval(instrument.Frequencies, powers)
        freq_maps = hp.ud_grade(freq_maps.T, NSIDE)
        s = np.linspace(1, 2, hp.nside2npix(NSIDE))
        np.random.shuffle(s)
        freq_maps = freq_maps * s

        res = basic_comp_sep(components, instrument,
                             freq_maps, nside=NSIDE_PARAM)
        aac(res.x.flatten(), powers, rtol=1e-5)
        aac(res.s.flatten(), s, rtol=1e-5)
        aaae(res.chi, 0, decimal=2)

    def test_T_s_2_param_0(self):
        NSIDE = 2
        NSIDE_PARAM = 0
        INSTRUMENT = 'test'

        np.random.seed(0)
        instrument = get_instrument(NSIDE, INSTRUMENT)
        components = [cm.Component('(nu - nu0)**power', nu0=30)]
        components[0].defaults = [1.]
        power = 1.5
        freq_maps = components[0].eval(instrument.Frequencies, power)
        s = np.linspace(1, 2, hp.nside2npix(NSIDE))
        np.random.shuffle(s)
        freq_maps = freq_maps[:, np.newaxis] * s

        res = basic_comp_sep(components, instrument,
                             freq_maps, nside=NSIDE_PARAM)
        aac(res.x.flatten(), power, rtol=1e-5)
        aac(res.s.flatten(), s, rtol=1e-5)
        aaae(res.chi, 0, decimal=2)

    def test_T_s_2_param_2(self):
        NSIDE = 2
        NSIDE_PARAM = 2
        INSTRUMENT = 'test'

        np.random.seed(0)
        instrument = get_instrument(NSIDE, INSTRUMENT)
        components = [cm.Component('(nu - nu0)**power', nu0=30)]
        components[0].defaults = [1.]
        powers = np.linspace(1, 2, hp.nside2npix(NSIDE_PARAM))
        np.random.shuffle(powers)
        freq_maps = components[0].eval(instrument.Frequencies, powers)
        freq_maps = hp.ud_grade(freq_maps.T, NSIDE)
        s = np.linspace(1, 2, hp.nside2npix(NSIDE))
        np.random.shuffle(s)
        freq_maps = freq_maps * s

        res = basic_comp_sep(components, instrument,
                             freq_maps, nside=NSIDE_PARAM)
        aac(res.x.flatten(), powers, rtol=1e-5)
        aac(res.s.flatten(), s, rtol=1e-5)
        aaae(res.chi, 0, decimal=2)

    def test_T_s_2_param_1_mask(self):
        NSIDE = 2
        NSIDE_PARAM = 1
        INSTRUMENT = 'test'
        GAL_CUT = np.radians(30)

        np.random.seed(0)
        instrument = get_instrument(NSIDE, INSTRUMENT)
        components = [cm.Component('(nu - nu0)**power', nu0=30)]
        components[0].defaults = [1.]
        powers = np.linspace(1, 2, hp.nside2npix(NSIDE_PARAM))
        np.random.shuffle(powers)
        freq_maps = components[0].eval(instrument.Frequencies, powers)
        freq_maps = hp.ud_grade(freq_maps.T, NSIDE)
        s = np.linspace(1, 2, hp.nside2npix(NSIDE))
        np.random.shuffle(s)
        freq_maps = freq_maps * s

        id_pix = np.arange(hp.nside2npix(NSIDE))
        mask = np.abs(hp.pix2vec(NSIDE, id_pix)[2]) < np.sin(GAL_CUT)
        freq_maps[..., mask] = hp.UNSEEN
        s[..., mask] = hp.UNSEEN

        res = basic_comp_sep(components, instrument,
                             freq_maps, nside=NSIDE_PARAM)

        s[..., mask] = hp.UNSEEN
        x_mask = hp.ud_grade(s, NSIDE_PARAM) == hp.UNSEEN
        powers[x_mask] = hp.UNSEEN
        aac(res.x.flatten(), powers, rtol=1e-5)
        aac(res.s.flatten(), s, rtol=1e-5)
        aaae(res.chi[..., mask], hp.UNSEEN, decimal=2)
        aaae(res.chi[..., ~mask], 0, decimal=2)


    def test_T_s_2_param_0_mask(self):
        print  'test_T_s_2_param_0_mask'
        NSIDE = 2
        NSIDE_PARAM = 0
        INSTRUMENT = 'test'
        GAL_CUT = np.radians(30)

        np.random.seed(0)
        instrument = get_instrument(NSIDE, INSTRUMENT)
        components = [cm.Component('(nu - nu0)**power', nu0=30)]
        components[0].defaults = [1.]
        power = 1.5
        freq_maps = components[0].eval(instrument.Frequencies, power)
        s = np.linspace(1, 2, hp.nside2npix(NSIDE))
        np.random.shuffle(s)
        freq_maps = freq_maps[:, np.newaxis] * s

        id_pix = np.arange(hp.nside2npix(NSIDE))
        mask = np.abs(hp.pix2vec(NSIDE, id_pix)[2]) < np.sin(GAL_CUT)
        freq_maps[..., mask] = hp.UNSEEN
        s[..., mask] = hp.UNSEEN

        res = basic_comp_sep(components, instrument,
                             freq_maps, nside=NSIDE_PARAM)
        aac(res.x.flatten(), power, rtol=1e-5)
        aac(res.s.flatten(), s, rtol=1e-5)
        aaae(res.chi[..., mask], hp.UNSEEN, decimal=2)
        aaae(res.chi[..., ~mask], 0, decimal=2)


if __name__ == '__main__':
    unittest.main()
