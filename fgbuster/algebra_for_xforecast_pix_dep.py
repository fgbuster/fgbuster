"""
Algebra around the spectral likelihood needed for the pixel dependent xForecast.
In particular:
- Pixel dependent 2nd derivative of the average spectral likelihood
- W, W_dB, W_dBdB
"""
import numpy as np
from scipy.linalg import lapack
from scipy.sparse import csc_matrix
from numpy.linalg import inv
from collections.abc import Iterable
from scipy.sparse import csc_matrix


def av_sp_lik_2nd_derivative(d, A, d_A, d_d_A, comp_of_dB, invN, sigB=None):
    """
    Function to implement Eq. (A5) of Stompor et al. (2016):
    2nd derivative of the average spectral likelihood
    d: noiseless data
    A: mixing matrix
    d_A: derivative of A wrt Beta
    d_d_A: 2nd derivative derivative of A wrt (Beta,Beta)
    comp_of_dB: list of non-zero components of the mixing matrix derivative, 
        and of the patch ids for each spectral parameter
        shape: n_beta_types x 2 (0 for the only non-zero comp, 1 for patch id array)
    invN: inverse of the noise covariance matrix
    patch_ids: list of arrays of patches ids (one array per spectral parameter)
        shape: n_beta_types x n_pix
    sigB: std dev of the Gaussian prior if any
        shape: n_beta_types

    Returns an array of the derivative with respect to each sp param
    """
    ## get the number of beta types (typically 3: Bd, Td, Bs)
    n_beta_types = len(comp_of_dB)

    ## compute derivatives independent objects
    ddt = get_ddt(d, invN)

    At_N = get_At_N(A, invN)
    Na = get_Na(At_N, A)
    N_A_Na = get_N_A_Na(invN, A, Na)
    P = get_P(invN, N_A_Na, At_N)

    d_d_L = []
    print('\n\nComputing the average spectral lik 2nd derivative...')
    for i in range(n_beta_types):
        for j in range(n_beta_types):
            if j > i:
                continue
            print(i, j)

            # if there is a Gaussian prior in the likelihood with std_dev sigB
            prior = 0
            if i==j:
                if isinstance(sigB, Iterable) and sigB[i] is not None:
                    prior = 2.0/(sigB[i]**2)

            ## compute the pix dependent quantities
            # stored as a single map, even if different pixels 
            # refer to different patches (hence betas)
            algebra_ij = get_algebra_ij(i, j, A, Na, N_A_Na, P, d_A, d_d_A, comp_of_dB)
            ## contraction of the 5 terms with A and N and <d dT> and Tr of all of it
            d_d_L_ij = np.einsum('pij, psji -> p', algebra_ij, ddt, optimize=True)
            ## sum over the pixels
            # get the smallest patches as the intersection of the two current sp params
            patch_ids_small = comp_of_dB[i][1] + comp_of_dB[j][1]*len(comp_of_dB[i][1])
            # got back to ordinal values
            patch_ids_small_unique_sorted = sorted(set(patch_ids_small))
            value_to_index = {value: index for index, value in enumerate(patch_ids_small_unique_sorted)}
            patch_ids_final = np.array([value_to_index[num] for num in patch_ids_small])
            # sum over the smallest patches
            d_d_L.append(np.bincount(patch_ids_final, weights=d_d_L_ij) - prior)

    return d_d_L


def weight_derivatives(A, d_A, comp_of_dB, invN, i_cmb):
    """
    Function to calculate the elements of the Taylor expansion of the weighting operator
    A: mixing matrix
    d_A: derivative of A wrt Beta
    comp_of_dB: list of non-zero components of the mixing matrix derivative,
        and of the patch ids for each spectral parameter
        shape: n_beta_types x 2 (0 for the only non-zero comp, 1 for patch id array)
    invN: inverse of the noise covariance matrix
    patch_ids: list of arrays of patches ids (one array per spectral parameter)
        shape: n_beta_types x n_pix

    Returns an array of the derivative with respect to each sp param
    """
    # get the number of beta types (typically 3: Bd. Td, Bs)
    n_beta_types = len(comp_of_dB)

    # compute derivatives independent objects
    At_N = get_At_N(A, invN) # A^T N^-1
    Na = get_Na(At_N, A) # (A^T N^-1 A)^-1
    N_A_Na = get_N_A_Na(invN, A, Na) # N^-1 A (A^T N^-1 A)^-1
    P = get_P(invN, N_A_Na, At_N) # P
    W = W_pix_dep(A, invN)[:,i_cmb,:]

    print('\n\nComputing the weights 1st derivatives...')
    W_dB = np.array([get_W_i(i, At_N, Na, N_A_Na, P, d_A, comp_of_dB)[:,i_cmb:i_cmb+1,:] for i in range(n_beta_types)])

    return W, W_dB


def list_to_matrix(d_d_L, comp_of_dB):
    """
    Function to re-format the list into a matrix for inversion
    """

    n_beta_types = len(comp_of_dB)

    k=0
    M = []
    for i in range(n_beta_types):

        M_tmp = []

        for j in range(n_beta_types):

            patch_ids_small = np.array((comp_of_dB[j][1], comp_of_dB[i][1]))
            patch_ids_small_unique = np.unique(patch_ids_small, axis=1)

            if j < i:
                M_tmptmp = np.zeros((np.max(patch_ids_small_unique[1]) + 1, np.max(patch_ids_small_unique[0]) + 1))
                M_tmptmp[patch_ids_small_unique[1], patch_ids_small_unique[0]] = d_d_L[k]
                M_tmp.append(M_tmptmp)
                k += 1
            elif j == i:
                M_tmptmp = np.zeros((np.max(patch_ids_small_unique[1]) + 1, np.max(patch_ids_small_unique[0]) + 1))
                M_tmptmp[patch_ids_small_unique[1], patch_ids_small_unique[0]] = d_d_L[k]
                M_tmp.append(0.5*M_tmptmp) # factor of 0.5 needed for symmetrization
                k += 1
            else:
                M_tmp.append(np.zeros((np.max(patch_ids_small_unique[1]) + 1, np.max(patch_ids_small_unique[0]) + 1)))
        
        M.append(M_tmp)


    matrix = np.block(M)
    matrix = matrix + matrix.T

    return matrix


def list_to_matrix_sparse(d_d_L, comp_of_dB, mask):
    """
    Function to re-format the list into a sparse matrix for the production of dbeta realizations
    """

    n_beta_types = len(comp_of_dB)

    k = 0
    row = []
    col = []
    offset = np.zeros(n_beta_types, dtype=np.int32)
    for i in range(n_beta_types):
        for j in range(n_beta_types):
            if j > i:
                continue

            if i == j: d_d_L[k] *= 0.5 # factor of 0.5 needed for symmetrization

            patch_ids_small_unique = little_patch_ij(comp_of_dB, j, i, mask)
            patch_ids_small_unique = patch_ids_small_unique[:, patch_ids_small_unique[0] >= 0] # negative patch_ids are the masked pixels

            if i < n_beta_types - 1: offset[i+1] = patch_ids_small_unique.shape[1]
            row.append(patch_ids_small_unique[0] + np.cumsum(offset)[j]) # offset needed because patch_ids always start from 0
            col.append(patch_ids_small_unique[1] + np.cumsum(offset)[i])

            k += 1

    d_d_L = np.concatenate(d_d_L)
    d_d_L = np.tile(d_d_L[d_d_L != 0], 2)
    row = np.concatenate(row)
    col = np.concatenate(col)
    row_tot = np.concatenate((row, col)) # Add col to row (and vice versa) to handle symmetrization
    col_tot = np.concatenate((col, row))

    matrix = csc_matrix((d_d_L, (row_tot, col_tot))) # csc_matrix is better for cholesky decomposition later on

    return matrix

def get_ddt(d, invN):
    # <d dT>

    return np.einsum('isp, jsp -> psij', d, d, optimize=True)# + inv(invN)


def get_At_N(A, invN):
    # A^T N^-1

    return np.einsum('pij, ik -> pjk', A, invN, optimize=True)


def get_Na(At_N, A):
    # Na = (A^T N^-1 A)^-1
    Na = np.einsum('pij, pjk -> pik', At_N, A, optimize=True)
    
    return inv(Na)


def get_N_A_Na(invN, A, Na):
    # N_A_Na = N^-1 A N_a

    return np.einsum('ij, pjk, pkl -> pil', invN, A, Na, optimize=True)


def get_P(invN, N_A_Na, At_N):
    """
    Projection operator
    (Eq. (8) of Stompor et al. (2016))
    """
    # P = N^-1 - N^-1 A N_a A^T N^-1

    return invN - np.einsum('pij, pjl -> pil', N_A_Na, At_N, optimize=True)


def get_algebra_ij(i, j, A, Na, N_A_Na, P, d_A, d_d_A, comp_of_dB):
    """
    Compute the algebra with A, d_A, d_d_A and N for the av sp lik 2nd derivative
    for a given pair of spectral parameters i and j.
    In Eq. (A5) of Stompor et al. (2016) it is the sum of the terms
    in the squared brackets before multiplying by <d dT>
    """
    # recurring objects
    dAt_i_P = get_dAt_P(i, P, d_A) # A^T_{, i} P
    if i == j: # A^T_{, j} P
        dAt_j_P = dAt_i_P
    else:
        dAt_j_P = get_dAt_P(j, P, d_A) 
    N_A_Na_dAt_j = np.einsum('pi, pj -> pij', N_A_Na[:,:,comp_of_dB[j][0]], d_A[j][:,:,0], optimize=True) # N^-1 A (A^T N^-1 A)^-1 A_{, j}

    # compute each term
    term1 = np.einsum('pi, p, pj -> pij', dAt_j_P, Na[:,comp_of_dB[j][0],comp_of_dB[i][0]], dAt_i_P, optimize=True) # P A_{, j} (A^T N^-1 A)^-1 A^T_{, i} P
    if len(d_d_A[i][j][:,0]) == 1:
        term2 = 0
    else:
        term2 = np.einsum('pi, pj, pjk -> pik', N_A_Na[:,:,comp_of_dB[i][0]], 
                          np.reshape(d_d_A[i][j][:,0], (A.shape[0], A.shape[1])), P, optimize=True) # N^-1 A (A^T N^-1 A)^-1 A^T_{, ij} P
    term3 = np.einsum('pij, pj, pk -> pik', N_A_Na_dAt_j, N_A_Na[:,:,comp_of_dB[i][0]], dAt_i_P, optimize=True) # N^-1 A (A^T N^-1 A)^-1 A_{, j} N^-1 A (A^T N^-1 A)^-1 A^T_{, i} P
    if i == j: 
        N_A_Na_dA_i = N_A_Na_dAt_j  # needed in term5
        term4 = term3
    else:
        #dAt_j_P = get_dAt_P(j, P, d_A)
        N_A_Na_dA_i = np.einsum('pi, pj -> pij', N_A_Na[:,:,comp_of_dB[i][0]], d_A[i][:,:,0], optimize=True) # N^-1 A (A^T N^-1 A)^-1 A^T_{, i}
        term4 = np.einsum('pij, pj, pk -> pik', N_A_Na_dA_i, N_A_Na[:,:,comp_of_dB[j][0]], dAt_j_P, optimize=True) # N^-1 A (A^T N^-1 A)^-1 A^T_{, i} N^-1 A (A^T N^-1 A)^-1 A^T_{, j} P
    term5 = np.einsum('pif, pf, pj -> pij', N_A_Na_dA_i, dAt_j_P, N_A_Na[:,:,comp_of_dB[j][0]], optimize=True) # N^-1 A (A^T N^-1 A)^-1 A^T_{, i} P A_{, j} (A^T N^-1 A)^-1 A^T N^-1

    # sum all the terms
    sum_all = term1 + term2 - term3 - term4 - term5

    return sum_all


def get_W_i(i, At_N, Na, N_A_Na, P, d_A, comp_of_dB):

    dAt_i_P = get_dAt_P(i, P, d_A) # A_{, i}^T P
    At_N_dA_i_Na_At_N = np.einsum('pij, pj, pk -> pik', At_N, d_A[i][:,:,0], N_A_Na[:,:,comp_of_dB[i][0]], optimize=True) # A^T N^-1 A_{, i} (A^T N^-1 A)^-1 A^T N^-1

    W_i = np.einsum('pi, pj -> pij', Na[:,:,comp_of_dB[i][0]], dAt_i_P) - np.einsum("pij, pjk -> pik", Na, At_N_dA_i_Na_At_N, optimize=True)

    return W_i


def get_W_ij(i, j, At_N, Na, N_A_Na, P, d_A, comp_of_dB):

    dAt_i_P = get_dAt_P(i, P, d_A) # A_{, i}^T P
    At_N_dA_i_Na_At_N = np.einsum('pij, pj, pk -> pik', At_N, d_A[i][:,:,0], N_A_Na[:,:,comp_of_dB[i][0]], optimize=True) # A^T N^-1 A_{, i} (A^T N^-1 A)^-1 A^T N^-1

    W_i = np.einsum('pi, pj -> pij', Na[:,:,comp_of_dB[i][0]], dAt_i_P, optimize=True) - np.einsum("pij, pjk -> pik", Na, At_N_dA_i_Na_At_N, optimize=True)

    return W_i


def get_dAt_P(i, P, d_A):
    # d_At P

    return np.einsum('pi, pki -> pk', d_A[i][:,:,0], P, optimize=True)


def W_pix_dep(A, invN):
    """
    W = (A^T N^-1 A)^-1 A^T N^-1
    """
    At_N = get_At_N(A, invN)
    Na = get_Na(At_N, A)

    return np.einsum('pij, pjk -> pik', Na, At_N, optimize=True)


def format_patch_id(max_id, patch_ids_unique, i):
    """
    Redefine patch_ids when the sky is masked
    Otherwise does nothing
    """
    for j in np.arange(max_id):
        if j > np.max(patch_ids_unique[i]): continue
        while not np.any(patch_ids_unique[i] == j): patch_ids_unique[i,patch_ids_unique[i] > j] -= 1

    return patch_ids_unique


def little_patch_ij(comp_of_dB, i, j, mask):
    """
    Defines the patches with unique (beta_i, beta_j)
    Account for the mask which is given id -1
    """
    patch_ids = np.array([comp_of_dB[i][1], comp_of_dB[j][1]]) # create vector of patch_ids maps for spectral params i and j
    patch_ids[..., mask==0] = -1 # all masked pixels are given patch_id -1

    patch_ids_unique = np.unique(patch_ids, axis=1) # keep only distinct vectors -> builds the little patches for params i and j
    patch_ids_unique = format_patch_id(np.max(comp_of_dB[i][1]), patch_ids_unique, 0)
    patch_ids_unique = format_patch_id(np.max(comp_of_dB[j][1]), patch_ids_unique, 1)

    return patch_ids_unique


def little_patch(comp_of_dB, mask, return_patch=False):
    """
    Defines the id of little patches with unique mixing matrix
    """

    n_beta_types = len(comp_of_dB)

    patch_ids = np.array([comp_of_dB[i][1] for i in range(n_beta_types)]) # create vector of patch_ids maps for all spectral params
    patch_ids[..., mask==0] = -1 # all masked pixels are given patch_id -1
    for i in range(n_beta_types): patch_ids = format_patch_id(np.max(comp_of_dB[i][1]), patch_ids, i)

    patch_ids_unique = np.unique(patch_ids, axis=1) # keep only distinct vectors -> builds the little patches

    if return_patch:
        return patch_ids_unique, patch_ids

    return patch_ids_unique


def little_patch_from_patch_ids(patch_ids):

    patch_ids_unique = np.unique(np.array(patch_ids), axis=1) # keep only distinct vectors -> builds the little patches

    return patch_ids_unique


def little_patch_ids_map(patch_ids, patch_ids_unique):
    """
    Map of little patches with new definition of little patches id
    """

    little_patch_ids = np.full(patch_ids.shape[-1], -1, dtype=np.int32)
    patch_ids_unique = patch_ids_unique[:, patch_ids_unique[0] >= 0]
    
    ind_trans = [np.argwhere(np.all(patch_ids == patch_ids_unique[:,i:i+1], axis=0)).flatten() for i in np.arange(patch_ids_unique.shape[-1])]
    for i in np.arange(len(ind_trans)):
        little_patch_ids[ind_trans[i]] = i

    return little_patch_ids


def param_to_patch(dbeta, patch_ids_unique):

    patch_ids_unique = patch_ids_unique[:, patch_ids_unique[0] >= 0]

    offset = np.insert(np.cumsum(np.max(patch_ids_unique[:-1,:]+1, axis=1)), 0, 0, axis=0)
    dbeta_patch = np.zeros((dbeta.shape[0],) + patch_ids_unique.shape)
    dbeta_patch = np.swapaxes(np.array([dbeta[:,patch_ids_unique[i,:]+offset[i]] for i in np.arange(patch_ids_unique.shape[0])]), 0, 1) # reshape delta beta as a (n_beta_types, n_little_patches) object

    return dbeta_patch


def patch_to_pixel(xpatch, little_patch_ids):

    xpix = np.array([xpatch[:,:,little_patch_ids[pix]] for pix in np.arange(little_patch_ids.shape[0])])

    return xpix


# Explicitly constructing the sqrt inverse of the Fisher matrix
def sqrtinv(Fisher):
    """
    Takes sparse Fisher matrix and returns the inverse of its upper triangular Cholesky decomposition matrix
    """
    try:
        from sksparse.cholmod import cholesky
    except ImportError:
        raise ImportError(
            "Required to install scikit-sparse!"
        )
    factor = cholesky(Fisher)
    L = (factor.L()).toarray()
    P = csc_matrix((np.ones(L.shape[0]), (np.arange(L.shape[0]), factor.P()))).toarray()
    print("Inverting Lt")
    Lt_inv, info = lapack.dtrtri(L.T)
    print("Uinv = Pt.Ltinv")
    Uinv = np.einsum("ji, jk -> ik", P, Lt_inv, optimize=True)

    return Uinv


# Generating realizations of delta beta
def dbeta(nreal, Fisher, comp_of_dB, mask, sparse=True, seed=42):

    patch_ids_unique, patch_ids = little_patch(comp_of_dB, mask, return_patch=True)
    little_patch_ids = little_patch_ids_map(patch_ids, patch_ids_unique)

    np.random.seed(seed)
    
    if sparse:
        size_dB = np.sum(np.array([np.max(patch_ids[i]+1) for i in np.arange(len(patch_ids))]))
        x = np.random.normal(size=(nreal, size_dB)) # centered reduced Gaussian realizations

        # Solving the linear system
        try:
            from sksparse.cholmod import cholesky
        except ImportError:
            raise ImportError(
                "Required to install scikit-sparse!"
            )
        factor = cholesky(Fisher)
        dbeta = np.array([factor.apply_Pt(factor.solve_Lt(x[i,:], use_LDLt_decomposition=False)) for i in np.arange(x.shape[0])])
    else:
        x = np.random.normal(size=(nreal, Fisher.shape[0]))

        Uinv = sqrtinv(Fisher)
        dbeta = np.einsum("ij, rj -> ri", Uinv, x)

    dbeta_patch = param_to_patch(dbeta, patch_ids_unique)
    dbeta_pix = patch_to_pixel(dbeta_patch, little_patch_ids)
    dbeta_pix = np.swapaxes(dbeta_pix, axis1=0, axis2=1)

    return dbeta_pix

