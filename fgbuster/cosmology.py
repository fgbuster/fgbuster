# FGBuster
# Copyright (C) 2019 Davide Poletti, Josquin Errard and the FGBuster developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Forecasting toolbox
"""
import os.path as op
import numpy as np
import pylab as pl
import healpy as hp
import scipy as sp
from .algebra import comp_sep, W_dBdB, W_dB, W, _mmm, _utmv, _mmv
from .mixingmatrix import MixingMatrix
from .separation_recipies import _force_keys_as_attributes


__all__ = [
    'xForecast',
]


CMB_CL_FILE = op.join(
     op.dirname(__file__), 'templates/Cls_Planck2018_%s.fits')


def xForecast(components, instrument, d_fgs, lmin, lmax,
              Alens=1.0, r=0.001, make_figure=False,
              **minimize_kwargs):
    """ xForecast

    Run XForcast (Stompor et al, 2016) using the provided instrumental
    specifications and input foregrounds maps. If the foreground maps match the
    components provided (constant spectral indices are assumed), it reduces to
    CMB4cast (Errard et al, 2011). Currently, only polarization is considered
    fot component separation and only the BB power spectrum for cosmological
    analysis.

    Parameters
    ----------
    components: list
         `Components` of the mixing matrix
    instrument: PySM.Instrument
        Instrument object used to define the mixing matrix and the
        frequency-dependent noise weight.
        It is required to have:

         - frequencies

        however, also the following are taken into account, if provided

         - sens_P (define the frequency inverse noise)

    d_fgs: ndarray
        The foreground maps. No CMB. Shape `(n_freq, n_stokes, n_pix)`.
        If some pixels have to be masked, set them to zero.
        Since (cross-)spectra of the maps will be computed, you might want to
        apodize your mask (use the same apodization for all the frequency).
    lmin: int
        minimum multipole entering the likelihood computation
    lmax: int
        maximum multipole entering the likelihood computation
    Alens: float
        Amplitude of the lensing B-modes entering the likelihood on r
    r: float
        tensor-to-scalar ratio assumed in the likelihood on r
    minimize_kwargs: dict
        Keyword arguments to be passed to `scipy.optimize.minimize` during
        the fitting of the spectral parameters.
        A good choice for most cases is
        `minimize_kwargs = {'tol': 1, options: {'disp': True}}`. `tol` depends
        on both the solver and your signal to noise: it should ensure that the
        difference between the best fit -logL and and the minimum is well less
        then 1, without exagereting (a difference of 1e-4 is useless).
        `disp` also triggers a verbose callback that monitors the convergence.

    Returns
    -------
    xFres: dict
        xForecast result. It includes

        - the fitted spectral parameters
        - noise-averaged post-component separation CMB power spectrum

          - noise spectrum
          - statistical residuals spectrum
          - systematic residuals spectrum

        - noise-averaged cosmological likelihood

    """
    # Preliminaries
    instrument = _force_keys_as_attributes(instrument)
    nside = hp.npix2nside(d_fgs.shape[-1])
    n_stokes = d_fgs.shape[1]
    n_freqs = d_fgs.shape[0]
    invN = np.diag(hp.nside2resol(nside, arcmin=True) / (instrument.Sens_P))**2
    mask = d_fgs[0, 0, :] != 0.
    fsky = mask.astype(float).sum() / mask.size
    ell = np.arange(lmin, lmax+1)
    print('fsky = ', fsky)

    ############################################################################
    # 1. Component separation using the noise-free foregrounds templare
    # grab the max-L spectra parameters with the associated error bars
    print('======= ESTIMATION OF SPECTRAL PARAMETERS =======')
    A = MixingMatrix(*components)
    A_ev = A.evaluator(instrument.Frequencies)
    A_dB_ev = A.diff_evaluator(instrument.Frequencies)

    x0 = np.array([x for c in components for x in c.defaults])
    if n_stokes == 3:  # if T and P were provided, extract P
        d_comp_sep = d_fgs[:, 1:, :]
    else:
        d_comp_sep = d_fgs

    res = comp_sep(A_ev, d_comp_sep.T, invN, A_dB_ev, A.comp_of_dB, x0,
                   **minimize_kwargs)

    res.params = A.params
    res.s = res.s.T
    A_maxL = A_ev(res.x)
    A_dB_maxL = A_dB_ev(res.x)
    A_dBdB_maxL = A.diff_diff_evaluator(instrument.Frequencies)(res.x)

    print('res.x = ', res.x)

    ############################################################################
    # 2. Estimate noise after component separation
    ### A^T N_ell^-1 A
    print('======= ESTIMATION OF NOISE AFTER COMP SEP =======')
    i_cmb = A.components.index('CMB')
    Cl_noise = _get_Cl_noise(instrument, A_maxL, lmax)[i_cmb, i_cmb, lmin:]

    ############################################################################
    # 3. Compute spectra of the input foregrounds maps
    ### TO DO: which size for Cl_fgs??? N_spec != 1 ? 
    print ('======= COMPUTATION OF CL_FGS =======')
    if n_stokes == 3:  
        d_spectra = d_fgs
    else:  # Only P is provided, add T for map2alm
        d_spectra = np.zeros((n_freqs, 3, d_fgs.shape[2]), dtype=d_fgs.dtype)
        d_spectra[:, 1:] = d_fgs

    # Compute cross-spectra
    almBs = [hp.map2alm(freq_map, lmax=lmax, iter=10)[2] for freq_map in d_spectra]
    Cl_fgs = np.zeros((n_freqs, n_freqs, lmax+1), dtype=d_fgs.dtype)
    for f1 in range(n_freqs):
        for f2 in range(n_freqs):
            if f1 > f2:
                Cl_fgs[f1, f2] = Cl_fgs[f2, f1]
            else:
                Cl_fgs[f1, f2] = hp.alm2cl(almBs[f1], almBs[f2], lmax=lmax)

    Cl_fgs = Cl_fgs[..., lmin:] / fsky

    ############################################################################
    # 4. Estimate the statistical and systematic foregrounds residuals
    print('======= ESTIMATION OF STAT AND SYS RESIDUALS =======')

    W_maxL = W(A_maxL, invN=invN)[i_cmb, :]
    W_dB_maxL = W_dB(A_maxL, A_dB_maxL, A.comp_of_dB, invN=invN)[:, i_cmb]
    W_dBdB_maxL = W_dBdB(A_maxL, A_dB_maxL, A_dBdB_maxL,
                         A.comp_of_dB, invN=invN)[:, :, i_cmb]
    V_maxL = np.einsum('ij,ij...->...', res.Sigma, W_dBdB_maxL)

    # Check dimentions
    assert ((n_freqs,) == W_maxL.shape == W_dB_maxL.shape[1:]
                       == W_dBdB_maxL.shape[2:] == V_maxL.shape)
    assert (len(res.params) == W_dB_maxL.shape[0] 
                            == W_dBdB_maxL.shape[0] == W_dBdB_maxL.shape[1])

    # elementary quantities defined in Stompor, Errard, Poletti (2016)
    Cl_xF = {}
    Cl_xF['yy'] = _utmv(W_maxL, Cl_fgs.T, W_maxL)  # (ell,)
    Cl_xF['YY'] = _mmm(W_dB_maxL, Cl_fgs.T, W_dB_maxL.T)  # (ell, param, param)
    Cl_xF['yz'] = _utmv(W_maxL, Cl_fgs.T, V_maxL )  # (ell,)
    Cl_xF['Yy'] = _mmv(W_dB_maxL, Cl_fgs.T, W_maxL)  # (ell, param)
    Cl_xF['Yz'] = _mmv(W_dB_maxL, Cl_fgs.T, V_maxL)  # (ell, param)

    # bias and statistical foregrounds residuals
    res.noise = Cl_noise
    res.bias = Cl_xF['yy'] + 2 * Cl_xF['yz']  # S16, Eq 23
    res.stat = np.einsum('ij, lij -> l', res.Sigma, Cl_xF['YY'])  # E11, Eq. 12
    res.var = res.stat**2 + 2 * np.einsum('li, ij, lj -> l', # S16, Eq. 28
                                          Cl_xF['Yy'], res.Sigma, Cl_xF['Yy'])

    ###############################################################################
    # 5. Plug into the cosmological likelihood
    print ('======= OPTIMIZATION OF COSMO LIKELIHOOD =======')
    Cl_fid = {}
    Cl_fid['BB'] = _get_Cl_cmb(Alens=Alens, r=r)[2][lmin:lmax+1]
    Cl_fid['BuBu'] = _get_Cl_cmb(Alens=0.0, r=1.0)[2][lmin:lmax+1]
    Cl_fid['BlBl'] = _get_Cl_cmb(Alens=1.0, r=0.0)[2][lmin:lmax+1]

    res.BB = Cl_fid['BB']*1.0
    res.BuBu = Cl_fid['BuBu']*1.0
    res.BlBl = Cl_fid['BlBl']*1.0
    res.ell = ell
    if make_figure:
        fig = pl.figure( figsize=(14,12), facecolor='w', edgecolor='k' )
        ax = pl.gca()
        left, bottom, width, height = [0.2, 0.2, 0.15, 0.2]
        ax0 = fig.add_axes([left, bottom, width, height])
        ax0.set_title(r'$\ell_{\min}=$'+str(lmin)+\
            r'$ \rightarrow \ell_{\max}=$'+str(lmax), fontsize=16)

        ax.loglog(ell, Cl_fid['BB'], color='DarkGray', linestyle='-', label='BB tot', linewidth=2.0)
        ax.loglog(ell, Cl_fid['BuBu']*r , color='DarkGray', linestyle='--', label='primordial BB for r='+str(r), linewidth=2.0)
        ax.loglog(ell, res.stat, 'DarkOrange', label='statistical residuals', linewidth=2.0)
        ax.loglog(ell, res.bias, 'DarkOrange', linestyle='--', label='systematic residuals', linewidth=2.0)
        ax.loglog(ell, res.noise, 'DarkBlue', linestyle='--', label='noise after component separation', linewidth=2.0)
        ax.legend()
        ax.set_xlabel('$\ell$', fontsize=20)
        ax.set_ylabel('$C_\ell$ [$\mu$K-arcmin]', fontsize=20)
        ax.set_xlim(lmin,lmax)

    ## 5.1. data 
    Cl_obs = Cl_fid['BB'] + Cl_noise
    dof = (2 * ell + 1) * fsky
    YY = Cl_xF['YY']
    tr_SigmaYY = np.einsum('ij, lji -> l', res.Sigma, YY)

    ## 5.2. modeling
    def cosmo_likelihood(r_):
        # S16, Appendix C
        Cl_model = Cl_fid['BlBl'] * Alens + Cl_fid['BuBu'] * r_ + Cl_noise
        dof_over_Cl = dof / Cl_model
        ## Eq. C3
        U = np.linalg.inv(res.Sigma_inv + np.dot(YY.T, dof_over_Cl))
        
        ## Eq. C9
        first_row = np.sum(dof_over_Cl * (
            Cl_obs * (1 - np.einsum('ij, lji -> l', U, YY) / Cl_model) 
            + tr_SigmaYY))
        second_row = - np.einsum(
            'l, m, ij, mjk, kf, lfi',
            dof_over_Cl, dof_over_Cl, U, YY, res.Sigma, YY)
        trCinvC = first_row + second_row
       
        ## Eq. C10
        first_row = np.sum(dof_over_Cl * (Cl_xF['yy'] + 2 * Cl_xF['yz']))
        ### Cyclicity + traspose of scalar + grouping terms -> trace becomes
        ### Yy_ell^T U (Yy + 2 Yz)_ell'
        trace = np.einsum('li, ij, mj -> lm',
                          Cl_xF['Yy'], U, Cl_xF['Yy'] + 2 * Cl_xF['Yz'])
        second_row = - _utmv(dof_over_Cl, trace, dof_over_Cl)
        trECinvC = first_row + second_row

        ## Eq. C12
        logdetC = np.sum(dof * np.log(Cl_model)) - np.log(np.linalg.det(U))

        # Cl_hat = Cl_obs + tr_SigmaYY

        ## Bringing things together
        return trCinvC + trECinvC + logdetC


    # Likelihood maximization
    r_grid = np.logspace(-5,0,num=500)
    logL = np.array([cosmo_likelihood(r_loc) for r_loc in r_grid])
    ind_r_min = np.argmin(logL)
    r0 = r_grid[ind_r_min]
    if ind_r_min == 0:
        bound_0 = 0.0
        bound_1 = r_grid[1]
        # pl.figure()
        # pl.semilogx(r_grid, logL, 'r-')
        # pl.show()
    elif ind_r_min == len(r_grid)-1:
        bound_0 = r_grid[-2]
        bound_1 = 1.0
        # pl.figure()
        # pl.semilogx(r_grid, logL, 'r-')
        # pl.show()
    else:
        bound_0 = r_grid[ind_r_min-1]
        bound_1 = r_grid[ind_r_min+1]
    print('bounds on r = ', bound_0, ' / ', bound_1)
    print('starting point = ', r0)
    res_Lr = sp.optimize.minimize(cosmo_likelihood, [r0], bounds=[(bound_0,bound_1)], **minimize_kwargs)
    print ('    ===>> fitted r = ', res_Lr['x'])

    print ('======= ESTIMATION OF SIGMA(R) =======')
    def sigma_r_computation_from_logL(r_loc):
        THRESHOLD = 1.00
        # THRESHOLD = 2.30 when two fitted parameters
        delta = np.abs( cosmo_likelihood(r_loc) - res_Lr['fun'] - THRESHOLD )
        # print r_loc, cosmo_likelihood(r_loc),  res_Lr['fun']
        return delta

    if res_Lr['x'] != 0.0:
        sr_grid = np.logspace(np.log10(res_Lr['x']), 0, num=25)
    else:
        sr_grid = np.logspace(-5,0,num=25)

    slogL = np.array([sigma_r_computation_from_logL(sr_loc) for sr_loc in sr_grid ])
    ind_sr_min = np.argmin(slogL)
    sr0 = sr_grid[ind_sr_min]
    print('ind_sr_min = ', ind_sr_min)
    print('sr_grid[ind_sr_min-1] = ', sr_grid[ind_sr_min-1])
    print('sr_grid[ind_sr_min+1] = ', sr_grid[ind_sr_min+1])
    print('sr_grid = ', sr_grid)
    if ind_sr_min == 0:
        print('case # 1')
        bound_0 = res_Lr['x']
        bound_1 = sr_grid[1]
    elif ind_sr_min == len(sr_grid)-1:
        print('case # 2')
        bound_0 = sr_grid[-2]
        bound_1 = 1.0
    else:
        print('case # 3')
        bound_0 = sr_grid[ind_sr_min-1]
        bound_1 = sr_grid[ind_sr_min+1]
    print('bounds on sigma(r) = ', bound_0, ' / ', bound_1)
    print('starting point = ', sr0)
    res_sr = sp.optimize.minimize(sigma_r_computation_from_logL, [sr0], bounds=[(bound_0,bound_1)], **minimize_kwargs)
    print ('    ===>> sigma(r) = ', res_sr['x'] -  res_Lr['x'])
    res.cosmo_params = {}
    res.cosmo_params['r'] = (res_Lr['x'], res_sr['x']- res_Lr['x'])


    ###############################################################################
    # 6. Produce figures
    if make_figure:
        print ('======= GRIDDING COSMO LIKELIHOOD =======')
        r_grid = np.logspace(-4,-1,num=500)
        logL = np.array([ cosmo_likelihood(r_loc) for r_loc in r_grid ])
        chi2 = logL - np.min(logL)
        ax0.semilogx( r_grid,  np.exp(-chi2), color='DarkOrange', linestyle='-', linewidth=2.0, alpha=0.8 )
        ax0.axvline(x=r, color='k', linestyle='--')
        ax0.set_ylabel(r'$\mathcal{L}(r)$', fontsize=20)
        ax0.set_xlabel(r'tensor-to-scalar ratio $r$', fontsize=20)
        pl.show()

    return res

def _get_Cl_cmb(Alens=1., r=0.):
    power_spectrum = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(CMB_CL_FILE
                                         %'unlensed_scalar_and_tensor_r1')[:,:4000]
    return power_spectrum


def _get_Cl_noise(instrument, A, lmax):
    bl = [hp.gauss_beam(np.radians(b/60.), lmax=lmax) for b in instrument.Beams]
    nl = (np.array(bl) / np.radians(instrument.Sens_P/60.)[:, np.newaxis])**2
    AtNA = np.einsum('fi, fl, fj -> lij', A, nl, A)
    inv_AtNA = np.linalg.inv(AtNA)
    return inv_AtNA.swapaxes(-3, -1)
