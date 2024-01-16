import numpy as np

# Reduced-state algorithm for the calculation of factors for matrix decomposition. Same functionality as pwr2decomp.
# Additional parameter mem for the list length
def decomp_pwr2_expo_reduced(target_mat, codebook_mat, S=3, mem=10, debug=False):
    # Catch errors on the inputs
    if target_mat.shape[0] != codebook_mat.shape[0]:
        raise ValueError('Target and codebook matrix dimension do not match')

    if target_mat.ndim > 2 or codebook_mat.ndim > 2:
        raise ValueError('Matrix dimensions not 2')

    if debug:
        print('Starting algorithm decomp_pwr2_expo, S=' + str(S))

    target_mat = np.copy(target_mat)
    codebook_mat = np.copy(codebook_mat)

    # Extract sizes of the input matrices
    N, K = target_mat.shape
    N1, K1 = codebook_mat.shape

    # If the number of additions per column is NOT specified, setting it to the number rows in the target matrix
    if S is None:
        S = N

    # Initialization of the wiring matrix
    wiring_mat = np.zeros((K1, K))
    # Copying the target matrix for minimization
    D = np.copy(target_mat)

    # Calculating the magnitudes of the columns in the codebook matrix (squared L2 norm)
    with np.errstate(divide='ignore'):  # Avoid runtime warnings when diving by zero (Dividing by zero returns inf)
        r = np.sum(np.power(codebook_mat, 2), axis=0)
        r = np.ones(r.shape) / r

    for m in range(K):  # Algorithm runs through each column in the target matrix
        # best_vectors = []
        best_dmin = np.inf * np.ones(mem)
        best_rhomin = np.inf * np.ones((mem, S))
        best_nmin = -np.ones((mem, S))
        best_residuals = np.inf * np.ones((mem, N))
        for z in range(int(S)):
            if z == 0:
                best_dmin, best_rhomin, best_nmin = best_fitting_vectors(D[:, m], codebook_mat, r, S, mem)
                for i in range(mem):  # Calculate the residual vectors for subsequent calculations
                    with np.errstate(invalid='ignore'):
                        best_residuals[i, :] = D[:, m] - (best_rhomin[i, 0] * codebook_mat[:, int(best_nmin[i, 0])])
            # elif 0 < z < int(S-1):
            elif 0 < z < int(S):
                best_dmin_list = []
                best_rhomin_list = []
                best_nmin_list = []
                for i in range(mem):  # check for every residual vector
                    best_dmin_tmp, best_rhomin_tmp, best_nmin_tmp = best_fitting_vectors(best_residuals[i, :], codebook_mat, r, S, mem, s_cur=z, rhomin_cur=best_rhomin[i], nmin_cur=best_nmin[i])
                    best_dmin_list.append(best_dmin_tmp)
                    best_rhomin_list.append(best_rhomin_tmp)
                    best_nmin_list.append(best_nmin_tmp)
                best_dmin, best_rhomin, best_nmin = expurgate(best_dmin_list, best_rhomin_list, best_nmin_list, mem, S)
                # Updating the residuals
                best_residuals = residual_vectors(D[:, m], best_rhomin, best_nmin, codebook_mat)
                if z == int(S-1):  # Update the wiring vector
                    totally_best_idx = np.argmin(best_dmin)
                    totally_best_rhomin = best_rhomin[totally_best_idx, :]
                    totally_best_nmin = best_nmin[totally_best_idx, :]
                    for i in range(S):
                        wiring_mat[int(totally_best_nmin[i]), m] += totally_best_rhomin[i]
            # elif z == int(S-1):
            else:
                raise OverflowError('Undefined value for variable z')
    return wiring_mat


def find_best_idx(dneu, best_dmin):
    ele_comp = dneu < best_dmin
    for i in range(best_dmin.size):
        if ele_comp[int(i)]:
            return i
    return ValueError('Undefined output')


def update_params(best_dmin, best_rhomin, best_nmin, dneu, rho, n, S, mem, s_cur, rhomin_cur, nmin_cur):
    idx = find_best_idx(dneu, best_dmin)
    # dmin
    best_dmin = np.insert(best_dmin, idx, dneu)
    best_dmin = np.delete(best_dmin, mem, axis=0)
    # rhomin
    # rhomin_tmp = np.zeros(S)
    rhomin_cur[s_cur] = rho
    best_rhomin = np.insert(best_rhomin, idx, rhomin_cur, axis=0)
    best_rhomin = np.delete(best_rhomin, mem, axis=0)
    # nmin
    # nmin_tmp = np.zeros(S)
    nmin_cur[s_cur] = n
    best_nmin = np.insert(best_nmin, idx, nmin_cur, axis=0)
    best_nmin = np.delete(best_nmin, mem, axis=0)
    return best_dmin, best_rhomin, best_nmin


def best_fitting_vectors(d, codebook_mat, r, S, mem, s_cur=0, rhomin_cur=None, nmin_cur=None):
    if np.any(rhomin_cur == None):
        rhomin_cur = np.zeros(S)
    if np.any(nmin_cur == None):
        nmin_cur = np.zeros(S)
    # Init output variables
    best_dmin = np.inf * np.ones(mem)
    best_rhomin = np.inf * np.ones((mem, S))
    best_nmin = -np.ones((mem, S))
    for n in range(codebook_mat.shape[1]):
        x = np.copy(d)
        y = codebook_mat[:, n]
        with np.errstate(invalid='ignore'):  # Suppress warnings when multiplying by inf
            rho = np.dot(y, x) * r[n]
        # Find the two closest quantized projections from the optimal value rho
        s = np.sign(rho)
        rho1 = s * (2 ** (np.floor(np.log2(s * rho))))
        rho2 = 2 * rho1
        # Find the distances to the target vectors from the quantized projections rho1 and rho2
        with np.errstate(invalid='ignore'):
            dneu1 = np.linalg.norm(x - (rho1 * y))
            dneu2 = np.linalg.norm(x - (rho2 * y))

        if np.any(dneu1 < best_dmin):
            best_dmin, best_rhomin, best_nmin = update_params(best_dmin, best_rhomin, best_nmin, dneu1, rho1, n, S, mem, s_cur, rhomin_cur, nmin_cur)
        if np.any(dneu2 < best_dmin):
            best_dmin, best_rhomin, best_nmin = update_params(best_dmin, best_rhomin, best_nmin, dneu2, rho2, n, S, mem, s_cur, rhomin_cur, nmin_cur)
    return best_dmin, best_rhomin, best_nmin


def expurgate(best_dmin_list, best_rhomin_list, best_nmin_list, mem, S):
    if not len(best_dmin_list) == len(best_rhomin_list) and len(best_rhomin_list) == len(best_nmin_list):
        raise ValueError('List lengths do not match')
    # Init output vars
    best_dmin = np.inf * np.ones(mem)
    best_rhomin = np.inf * np.ones((mem, S))
    best_nmin = -np.ones((mem, S))

    best_dmin_ext = best_dmin_list[0]
    best_rhomin_ext = best_rhomin_list[0]
    best_nmin_ext = best_nmin_list[0]
    for l in range(1, len(best_dmin_list)):  # Merge into one large list
        best_dmin_ext = np.append(best_dmin_ext, best_dmin_list[l], axis=0)
        best_rhomin_ext = np.append(best_rhomin_ext, best_rhomin_list[l], axis=0)
        best_nmin_ext = np.append(best_nmin_ext, best_nmin_list[l], axis=0)
    for m in range(mem):  # Only keep the best mem results in the putput list
        idx_min = np.argmin(best_dmin_ext)
        best_dmin[m] = best_dmin_ext[idx_min]
        best_rhomin[m, :] = best_rhomin_ext[idx_min, :]
        best_nmin[m, :] = best_nmin_ext[idx_min, :]
        # Remove the currently selected idx from the sets
        best_dmin_ext = np.delete(best_dmin_ext, idx_min, axis=0)
        best_rhomin_ext = np.delete(best_rhomin_ext, idx_min, axis=0)
        best_nmin_ext = np.delete(best_nmin_ext, idx_min, axis=0)
    return best_dmin, best_rhomin, best_nmin


def residual_vectors(d, best_rhomin, best_nmin, codebook_mat):
    mem = best_rhomin.shape[0]
    S = best_nmin.shape[1]
    best_residuals = np.zeros((mem, d.size))
    for i in range(mem):
        for s in range(S):
            if s == 0:
                best_residuals[i, :] = d - (best_rhomin[i, s] * codebook_mat[:, int(best_nmin[i, s])])
            elif s > 0:
                best_residuals[i, :] = best_residuals[i, :] - (best_rhomin[i, s] * codebook_mat[:, int(best_nmin[i, s])])
            else:
                raise ValueError('Undefined Value for s')
    return best_residuals
