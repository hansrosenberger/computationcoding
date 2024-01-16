import numpy as np
import wiring_red as wiring_red_alg
from metrics import no_add, distortion


# Calculates the sparse matrix factor given a target matrix, a codebook matrix and a fixed number of non-zero entries
# per column
# @args: target_mat: NxK target matrix, i.e. the matrix to be approximated, expected as a numpy array; codebook_mat:
# codebook matrix, NxK numpy array, used for the approximation of the target matrix; S: number of non-zero elements per
# column in the wiring_matrix, i.e. the matrix approximation factor
# @returns: wiring_mat: KxK numpy array as the matrix factor with non-zeros entries constrained to signed power of two
def power2decomp(target_mat, codebook_mat, S, diff_signs=False, debug=False, pos_signs=False):
    if target_mat.shape[0] != codebook_mat.shape[0]:
        raise ValueError('Target and codebook matrix dimension do not match')

    if target_mat.ndim > 2 or codebook_mat.ndim > 2:
        raise ValueError('Matrix dimension greater than 2')

    # Extract sizes of the input matrices
    N, K = target_mat.shape
    N1, K1 = codebook_mat.shape

    if S is None:
        S = 2       # Set no of nonzero coefficients per col to 2 if not specified

    wiring_mat = np.zeros((K1, K))  # Initialization of the wiring matrix
    D = np.copy(target_mat)  # Copy of target matrix for minimization

    # Calculating the magnitudes of the columns in the codebook matrix (squared L2 norm)
    with np.errstate(divide='ignore'):  # Avoid runtime warnings when diving by zero (Dividing by zero returns inf)
        r = np.sum(np.power(codebook_mat, 2), axis=0)
        r = np.ones(r.shape)/r

    for m in range(K):  # Algorithm runs through each column in the target matrix
        dmin = np.Inf   # Initialize the minimum distance with inf
        rhomin = np.Inf  # Same goes for the optimum projection factor
        for z in range(int(S)):  # Loop through the number of nonzero elements
            if dmin > 0:
                nmin = -1  # Variable for the best index in the current loop
                for n in range(K1):  # Loop through all the columns in the codebook matrix
                    x = D[:, m]
                    y = codebook_mat[:, n]
                    with np.errstate(invalid='ignore'):  # Suppress warnings when multiplying by inf
                        rho = np.dot(y, x) * r[n]  # Find the best projection of the current codebook vector on the current target vector

                    # Find the two closest quantized projections from the optimal value rho
                    s = np.sign(rho)
                    rho1 = s * (2 ** (np.floor(np.log2(s * rho))))
                    rho2 = 2 * rho1
                    # Find the distances to the target vectors from the quantized projections rho1 and rho2
                    dneu1 = np.linalg.norm(x - (rho1 * y))
                    dneu2 = np.linalg.norm(x - (rho2 * y))

                    if z >= 1 and diff_signs and s == -1 and np.all(np.sign(wiring_mat[np.nonzero(wiring_mat[:, m]), m]) == -1):
                        pass
                    elif pos_signs and s == -1:
                        pass
                    else:
                        if dneu1 < dmin:
                            dmin, nmin, rhomin = dneu1, n, rho1
                        if dneu2 < dmin:
                            dmin, nmin, rhomin = dneu2, n, rho2

                # Update the wiring matrix and D for the best index (if one was found)
                if nmin >= 0:
                    wiring_mat[nmin, m] = wiring_mat[nmin, m] + rhomin
                    D[:, m] = D[:, m] - rhomin * codebook_mat[:, nmin]

                    if debug:  # Debugging
                        print('W update success: ', nmin, m, rhomin)
                        print('W: ', wiring_mat)

                else:
                    pass  # updating W failed
    return wiring_mat


# @args: target_mat: NxK target matrix, i.e. the matrix to be approximated, expected as a numpy array; codebook_mat:
# codebook matrix, NxK numpy array, used for the approximation of the target matrix; S: number of non-zero elements per
# column in the wiring_matrix, i.e. the matrix approximation factor; no_mat: Number of matrix factors desired;
# wiring_red: If True dead-end removal is enabled
# @returns: wiring_mat: KxKxno_mat numpy array as the matrix factor with non-zeros entries constrained to signed powers
# of two; P: approximated NxK matrix; add_cum: cumulative number of additions for all matrix factors;
# SQNR: Signal-to-Quantization-Noise-Ratio of the approximation
def decomp_pwr2_metrics(target_mat, codebook_mat_init, S, no_mat, SQNR_tgt=None, transpose=True, diff_signs=False, wiring_red=True, pos_signs=False):
    N, K = target_mat.shape
    if transpose:
        W = np.zeros((N, N, no_mat))
    else:
        W = np.zeros((K, K, no_mat))
    add_cum = 0
    target_mat = np.copy(target_mat)
    codebook_mat = np.copy(codebook_mat_init)
    if SQNR_tgt is None:
        for mat in range(no_mat):
            if transpose:
                W[:, :, mat] = power2decomp(target_mat.T, codebook_mat.T, S, diff_signs=diff_signs, pos_signs=pos_signs).T
                codebook_mat = W[:, :, mat] @ codebook_mat
            else:
                W[:, :, mat] = power2decomp(target_mat, codebook_mat, S, diff_signs=diff_signs, pos_signs=pos_signs)
                codebook_mat = codebook_mat @ W[:, :, mat]
    else:
        mat = 0
        while distortion.SNRmat(target_mat, codebook_mat) <= SQNR_tgt:
            if transpose:
                W[:, :, mat] = power2decomp(target_mat.T, codebook_mat.T, S, diff_signs=diff_signs, pos_signs=pos_signs).T
                codebook_mat = W[:, :, mat] @ codebook_mat
            else:
                W[:, :, mat] = power2decomp(target_mat, codebook_mat, S, diff_signs=diff_signs, pos_signs=pos_signs)
                codebook_mat = codebook_mat @ W[:, :, mat]
            mat += 1
            if mat == no_mat:  # Exit condition when W is filled
                break
        if mat <= no_mat:
            W = np.delete(W, np.s_[mat:no_mat], axis=2)  # Remove unused all zero matrices

    if wiring_red:
        W = wiring_red_alg.wiring_red(W, list=False)

    codebook_mat = np.copy(codebook_mat_init)
    for mat in range(W.shape[2]):
        if transpose:
            codebook_mat = W[:, :, mat] @ codebook_mat
            add_cum += no_add.no_add_MPA(W[:, :, mat])
        else:
            codebook_mat = codebook_mat @ W[:, :, mat]
            add_cum += no_add.no_add_MPA(W[:, :, mat])
    P = codebook_mat
    SQNR = distortion.SNRmat(target_mat, codebook_mat)
    return W, P, add_cum, SQNR


# Same functionality as power2decomp, accepts however an additional length array as input (required for path constrained
# algorithms)
def power2decomp2d_lz_length(target_mat, codebook_mat, S, length, max_length, debug=False):
    if target_mat.shape[0] != codebook_mat.shape[0]:
        raise ValueError('Target and codebook matrix dimension do not match')

    if target_mat.ndim > 2 or codebook_mat.ndim > 2:
        raise ValueError('Matrix dimension greater than 2')

    # Checking the extra parameters
    if length.shape[0] != codebook_mat.shape[1]:
        raise ValueError('Length array does not match the number of codewords')

    # Extract sizes of the input matrices
    N, K = target_mat.shape
    N1, K1 = codebook_mat.shape

    # If the number of additions per column is NOT specified, setting it to 2
    if S is None:
        S = 2

    # Initialization of the wiring matrix
    wiring_mat = np.zeros((K1, K))
    # Copying the target matrix for minimization
    D = np.copy(target_mat)

    # Calculating the magnitudes of the columns in the codebook matrix (squared L2 norm)
    with np.errstate(divide='ignore'):  # Avoid runtime warnings when diving by zero (Dividing by zero returns inf)
        r = np.sum(np.power(codebook_mat, 2), axis=0)
        r = np.ones(r.shape)/r

    for m in range(K):  # Algorithm runs through each column in the target matrix
        dmin = np.Inf   # Initialize the minimum distance with inf
        rhomin = np.Inf  # Same goes for the optimum projection factor
        lengths_cols = np.zeros(int(S))  # lengths for each column of the selected codeword
        for z in range(int(S)):  # Loop for the number of additions
            if dmin > 0:  # Check if the current minimum distance (dmin) is greater than zero (should not happen
                          # otherwise an exception is raised)
                nmin = -1  # Variable for the best index in the current loop
                cur_length = -1  # Running variable for the current length of the best selected codeword
                for n in range(K1):  # Loop through all the columns in the codebook matrix
                    if debug:  # Debugging
                        print('!m,z,n!', m, z, n)

                    # Skip columns where the length to the first selected codeword is too large
                    if z > 0 and np.abs(lengths_cols[0] - length[n]) > max_length:
                        continue

                    x = D[:, m]
                    y = codebook_mat[:, n]
                    with np.errstate(invalid='ignore'):  # Suppress warnings when multiplying by inf
                        rho = np.dot(y, x) * r[n]  # Find the best projection of the current codebook vector on the current target vector
                    # Find the two closest quantized projections from the optimal value rho
                    s = np.sign(rho)
                    rho1 = s * (2 ** (np.floor(np.log2(s*rho))))
                    rho2 = 2 * rho1
                    # Find the distances to the target vectors from the quantized projections rho1 and rho2
                    dneu1 = np.linalg.norm(x-(rho1*y))
                    dneu2 = np.linalg.norm(x-(rho2*y))

                    # Test if the quantized projections of the current vectors have a smaller error than previously
                    if dneu1 < dmin:
                        dmin, nmin, rhomin, cur_length = dneu1, n, rho1, length[n]

                    if dneu2 < dmin:
                        dmin, nmin, rhomin, cur_length = dneu2, n, rho2, length[n]

                # Update the wiring matrix and D for the best index (if one was found)
                if nmin >= 0:
                    wiring_mat[nmin, m] = wiring_mat[nmin, m] + rhomin
                    D[:, m] = D[:, m] - rhomin * codebook_mat[:, nmin]
                    lengths_cols[z] = cur_length
                else:
                    pass
    return wiring_mat
