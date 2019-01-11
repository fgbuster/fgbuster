#!/usr/bin/env python
from itertools import product
import unittest
from parameterized import parameterized
import numpy as np
from numpy.testing import assert_allclose as aac
from scipy.stats import kstest
import healpy as hp
import pysm
from fgbuster.algebra import _mv
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import get_instrument
import fgbuster.component_model as cm
from fgbuster.separation_recipies import (basic_comp_sep, weighted_comp_sep,
                                          _force_keys_as_attributes)


def _get_n_stokes(tag):
    if tag in 'IN':
        return 1
    elif tag == 'P':
        return 2
    else:
        raise ValueError('Unsupported tag: %s'%tag)


def _get_nside(tag):
    return int(tag.split('_')[1])


def _get_component(tag):
    if '_' in tag:
        components = sum([_get_component(c) for c in tag.split('_')], [])
    elif tag == 'fixedpowerlaw':
        components = [cm.PowerLaw(nu0=30, beta_pl=1.1)]
    elif tag == 'powerlaw':
        components =  [cm.PowerLaw(nu0=30)]
        components[0].defaults = [1.5]
    elif tag == 'curvedpowerlaw':
        components = [cm.PowerLaw(nu0=30, nu_pivot=40, running=None)]
        components[0].defaults = [-1.5, 1.8]
    else:
        raise ValueError('Unsupported tag: %s'%tag)
    return components


def _get_mask(tag, nside):
    if tag == 'nomask':
        return np.zeros(hp.nside2npix(nside)).astype(bool)
    elif 'gal' in tag:
        gal_cut = int(tag.split('gal')[-1])
        pix = np.arange(hp.nside2npix(nside))
        return np.abs(hp.pix2vec(nside, pix)[2]) < np.sin(np.radians(gal_cut))
    else:
        raise ValueError('Unsupported tag: %s'%tag)



def _get_instrument(tag, nside=None):
    if 'dict' in tag:
        instrument = {}
        instrument['Frequencies'] = np.arange(10., 300, 30.)
        if 'homo' in tag:
            instrument['Sens_I'] = (np.linspace(20., 40., 10) - 30)**2
            instrument['Sens_P'] = instrument['Sens_I']
        elif 'vary' in tag:
            np.random.seed(0)
            instrument['Cov_N'] = (np.linspace(20., 40., 10) - 30)**4
            instrument['Cov_N'] /= hp.nside2resol(nside, arcmin=True)**2
            shape = (instrument['Frequencies'].size, hp.nside2npix(nside))
            factor = 10**np.random.uniform(-1, 1, size=np.prod(shape))
            factor = factor.reshape(shape)
            instrument['Cov_N'] = instrument['Cov_N'][:, np.newaxis] * factor

            instrument['Cov_I'] = instrument['Cov_N'][:, np.newaxis]

            instrument['Cov_P'] = np.stack([instrument['Cov_N']]*2, axis=1)
    elif 'pysm' in tag:
        instrument = pysm.Instrument(get_instrument('test', nside))
    else:
        raise ValueError('Unsupported tag: %s'%tag)
    return instrument


def _get_sky(tag):
    np.random.seed(0)

    stokes, nside, nsidepar, components, mask, instrument = tag.split('__')
    n_stokes = _get_n_stokes(stokes)
    nside = _get_nside(nside)
    nsidepar = _get_nside(nsidepar)
    components = _get_component(components)
    mask = _get_mask(mask, nside)
    instrument = _get_instrument(instrument, nside)
    try:
        freqs = instrument.Frequencies
    except AttributeError:
        freqs = instrument['Frequencies']

    x0 = [x for c in components for x in c.defaults]
    if nsidepar and len(x0):
        for i in range(len(x0)):
            factor = np.linspace(0.8, 1.2, hp.nside2npix(nsidepar))
            np.random.shuffle(factor)
            x0[i] = x0[i] * factor
        ux0 = [hp.ud_grade(x0_i, nside) for x0_i in x0]
        A = MixingMatrix(*components).eval(freqs, *ux0)
        if stokes in 'IP':
            A = A[:, np.newaxis]
    else:
        A = MixingMatrix(*components).eval(freqs, *x0)
    x0 = np.array(x0)

    n_pix = hp.nside2npix(nside)
    n_comp = len(components)
    shape = (n_pix, n_comp) if stokes == 'N' else (n_pix, n_stokes, n_comp)
    s = np.linspace(10., 20., n_pix * n_stokes * n_comp)
    np.random.shuffle(s)
    s = s.reshape(shape)

    data = _mv(A, s)

    data[mask] = hp.UNSEEN
    s[mask] = hp.UNSEEN
    if nsidepar and len(x0):
        x_mask = hp.ud_grade(mask.astype(float), nsidepar) == 1.
        x0[..., x_mask] = hp.UNSEEN

    return data.T, s.T, x0


def _make_tag(stokes, nside, nsidepar, components, mask, instrument):
    sky_tag = '%s__nside_%i__nsidepar_%i__%s__%s__%s' % (
        stokes, nside, nsidepar, components, mask, instrument)
    comp_sep_tag = '%s__%s__nsidepar_%s' % (
        components, instrument, nsidepar)
    return '___%s___%s' % (sky_tag, comp_sep_tag)


class TestBasicCompSep(unittest.TestCase):
    stokess = 'IPN'
    nsides = [2]
    nsidepars = [0, 1, 2]
    componentss = ['powerlaw_curvedpowerlaw', 'fixedpowerlaw',
                   'fixedpowerlaw_powerlaw', 'curvedpowerlaw']
    masks = ['nomask', 'maskgal30']
    instruments = ['dict_homo', 'pysm']

    tags = []
    tags += [_make_tag(*args) for args in product(stokess[:], nsides[:1],
                                                  nsidepars[:1], componentss[:1],
                                                  masks[:1], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:],
                                                  nsidepars[:1], componentss[:1],
                                                  masks[:1], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:1],
                                                  nsidepars[:], componentss[:1],
                                                  masks[:1], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:1],
                                                  nsidepars[:1], componentss[:],
                                                  masks[:1], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:1],
                                                  nsidepars[:1], componentss[:1],
                                                  masks[:], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:1],
                                                  nsidepars[:1], componentss[:1],
                                                  masks[:1], instruments[:])]

    @parameterized.expand(tags)
    def test(self, tag):
        _, sky_tag, comp_sep_tag = tag.split('___')

        data, s, x = _get_sky(sky_tag)

        components, instrument, nsidepar = comp_sep_tag.split('__')
        components = _get_component(components)
        for c in components:
            c.defaults = [1.1 * d for d in c.defaults]

        instrument = _get_instrument(instrument)
        nsidepar = _get_nside(nsidepar)
        res = basic_comp_sep(components, instrument, data, nsidepar)

        if len(x):
            aac(res.x, x, rtol=1e-5)

        aac(res.s, s, rtol=1e-4)
        aac(res.chi[data == hp.UNSEEN], hp.UNSEEN, rtol=0)
        aac(res.chi[data != hp.UNSEEN], 0, atol=0.05)


class TestWeightedCompSep(unittest.TestCase):
    stokess = 'IPN'
    nsides = [2]
    nsidepars = [0, 1, 2]
    componentss = ['powerlaw_curvedpowerlaw', 'fixedpowerlaw',
                   'fixedpowerlaw_powerlaw', 'curvedpowerlaw']
    masks = ['nomask', 'maskgal30']
    instruments = ['dict_vary', 'dict_homo', 'pysm']

    tags = []
    tags += [_make_tag(*args) for args in product(stokess[:], nsides[:1],
                                                  nsidepars[:1], componentss[:1],
                                                  masks[:1], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:],
                                                  nsidepars[:1], componentss[:1],
                                                  masks[:1], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:1],
                                                  nsidepars[:], componentss[:1],
                                                  masks[:1], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:1],
                                                  nsidepars[:1], componentss[:],
                                                  masks[:1], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:1],
                                                  nsidepars[:1], componentss[:1],
                                                  masks[:], instruments[:1])]
    tags += [_make_tag(*args) for args in product(stokess[:1], nsides[:1],
                                                  nsidepars[:1], componentss[:1],
                                                  masks[:1], instruments[:])]

    @parameterized.expand(tags)
    def test(self, tag):
        _, sky_tag, comp_sep_tag = tag.split('___')

        data, s, x = _get_sky(sky_tag)

        components, instrument, nsidepar = comp_sep_tag.split('__')
        components = _get_component(components)
        for c in components:
            c.defaults = [1.1 * d for d in c.defaults]

        nside = hp.get_nside(data[0])
        instrument = _get_instrument(instrument, nside)
        nsidepar = _get_nside(nsidepar)
        cov = self._get_cov(instrument, sky_tag.split('__')[0], nside)
        res = weighted_comp_sep(components, instrument, data, cov, nsidepar)

        if len(x):
            aac(res.x, x, rtol=1e-5)

        aac(res.s, s, rtol=1e-4)
        aac(res.chi[data == hp.UNSEEN], hp.UNSEEN, rtol=0)
        aac(res.chi[data != hp.UNSEEN], 0, atol=0.05)

    @parameterized.expand(['%s_%s' % x for x in product(stokess, masks)])
    def test_weighting(self, tag):
        stokes, mask = tag.split('_')
        instrument = 'dict_vary'
        components = 'fixedpowerlaw'
        nside = 32
        sky_tag = '%s__nside_%i__nsidepar_%i__%s__%s__%s' % (
            stokes, nside, 0, components, mask, instrument)
        data, _, _ = _get_sky(sky_tag)
        instrument = _get_instrument(instrument, nside)
        components = _get_component(components)

        cov = self._get_cov(instrument, sky_tag.split('__')[0])
        np.random.seed(0)
        data += np.random.normal(size=cov.size).reshape(cov.shape) * cov**0.5
        res = weighted_comp_sep(components, instrument, data, cov)

        mask = hp.ma(data[0]).mask
        chi2s = (res.chi[:, ~mask]**2).sum(axis=0).flatten()
        dof = data.shape[0] - 1
        _, p_value = kstest(chi2s.flatten(), 'chi2', (dof,))
        assert p_value > 0.10


    def _get_cov(self, instrument, stokes, nside=None):
        instrument = _force_keys_as_attributes(instrument)
        cov_tag = 'Cov_%s' % stokes
        try:
            return getattr(instrument, cov_tag)
        except AttributeError:
            shape = instrument.Frequencies.shape
            if stokes == 'P':
                shape += (2,)
            elif stokes == 'I':
                shape += (1,)
            shape += (hp.nside2npix(nside),)
            if stokes == 'P':
                return (instrument.Sens_P[:, np.newaxis, np.newaxis]
                        * np.full(shape, 1./hp.nside2resol(nside, True)**2))
            elif stokes == 'I':
                return (instrument.Sens_I[:, np.newaxis, np.newaxis]
                        * np.full(shape, 1./hp.nside2resol(nside, True)**2))
            elif stokes == 'N':
                return (instrument.Sens_I[:, np.newaxis]
                        * np.full(shape, 1./hp.nside2resol(nside, True)**2))
            else:
                raise ValueError(stokes)


if __name__ == '__main__':
    unittest.main()
    '''
    suite = unittest.TestSuite()
    for method in dir(TestWeightedCompSep):
        if 'the failing thest' in method:
            suite.addTest(TestWeightedCompSep(method))
    unittest.TextTestRunner().run(suite)
    '''
