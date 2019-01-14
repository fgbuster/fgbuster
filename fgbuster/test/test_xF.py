#!/usr/bin/env python
import unittest
import numpy as np
from numpy.testing import assert_allclose as aac
import pysm
from fgbuster.observation_helpers import get_instrument, get_sky
from fgbuster import xForecast, CMB, Dust, Synchrotron
from .test_end2end import suppress_stdout

class TestXfCompSep(unittest.TestCase):

    def test_d0s0(self):

        # EXTERNAL_xFORECAST_RUN RESULTS
        EXT_BETA = [ 1.54000015, 19.99999402, -3.00000035]
        EXT_SIGMA_BETA =  np.array([[1.07107731e+09, 1.02802459e+07, 3.05900728e+07],
                                    [1.02802459e+07, 8.80751745e+05, 2.64258857e+05],
                                    [3.05900728e+07, 2.64258857e+05, 8.80649611e+05]])
        EXT_NOISE_POST_COMP_SEP = [6.985360419978725e-07, 6.954968812329504e-07, 6.952971018678473e-07, 6.97621370087622e-07, 6.97915989771863e-07, 6.981532542428136e-07, 6.984690244398313e-07, 6.963706038663776e-07, 6.962958090983174e-07, 6.999793141962897e-07, 6.966029199088166e-07, 6.998332244730213e-07, 6.97245540450936e-07, 7.013469190449905e-07, 6.98145319051069e-07, 6.997552902541847e-07, 7.006828378883164e-07, 6.993357502111902e-07, 7.016843277673384e-07, 7.02276431905913e-07, 7.009651598790946e-07, 7.024327502574484e-07, 7.058590396724249e-07, 7.035637090541009e-07, 7.034402740635456e-07, 7.05326337473677e-07, 7.086905417607417e-07, 7.067287662339356e-07, 7.06396320822362e-07, 7.075857215168964e-07, 7.102089978543108e-07, 7.118461226661247e-07]
        EXT_CL_STAT_RES = [3.43065437e-08, 2.13752688e-07, 2.50160994e-08, 4.39734801e-08, 1.75192647e-08, 2.10382699e-08, 9.55361360e-09, 8.80726572e-09, 7.34671936e-09, 4.24354505e-09, 3.50430309e-09, 3.21803173e-09, 3.62342203e-09, 1.83222822e-09, 2.40687985e-09, 1.76806752e-09, 2.57252032e-09, 1.19987889e-09, 1.71606507e-09, 1.01867261e-09, 1.11709059e-09, 1.05584166e-09, 8.37499498e-10, 1.04610499e-09, 7.27953346e-10, 7.55604710e-10, 5.50190292e-10, 6.38657310e-10, 4.82912230e-10, 5.21029442e-10, 4.77954181e-10]
        EXT_BIAS_R = 0.00100556
        EXT_SIGMA_R = 0.0003163

        nside = 16
        # define sky and foregrounds simulations
        sky = pysm.Sky(get_sky(nside, 'd0s0'))
        # define instrument
        instrument = pysm.Instrument(get_instrument('litebird', nside))
        # get noiseless frequency maps
        with suppress_stdout():
            freq_maps = instrument.observe(sky, write_outputs=False)[0]
        # take only the Q and U maps
        freq_maps = freq_maps[:,1:]
        # define components used in the modeling
        components = [CMB(), Dust(150.), Synchrotron(150.)]
        # call for xForecast 
        with suppress_stdout():
            res = xForecast(components, instrument, freq_maps, 2, 2*nside-1, 1.0, make_figure=False)
        # list of checks
        aac(EXT_BETA, res.x, rtol=1e-03)
        aac(np.diag(EXT_SIGMA_BETA), np.diag(res.Sigma_inv), rtol=5e-02)
        aac(EXT_NOISE_POST_COMP_SEP[0], res.noise[0], rtol=1e-02)
        aac(EXT_BIAS_R, res.cosmo_params['r'][0][0], rtol=1e-02)
        aac(EXT_SIGMA_R, res.cosmo_params['r'][1][0], rtol=1e-02)

if __name__ == '__main__':
    unittest.main()
