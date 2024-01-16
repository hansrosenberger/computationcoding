import numpy as np

import dmp


# Mixed algorithm: Path constrained version of fastlzdecomp in lz.py. max_length_diff controls the maximum depth
# difference of two nodes within the decomposition graph. length_penalty enables/disables the multiplicative penalty
# factor to the objective function (mse).
def fastlzdecomp_pathopt(A, SNRdBmin, mu=1, max_length_diff=np.Inf, full_W=False, length_penalty=True, max_add=np.Inf, debug=False):
    N, K = A.shape

    P = np.eye(K, K)
    Pj = dmp.power2decomp(np.copy(A).T, np.copy(P).T, 1).T
    nmin = np.linalg.norm(A - Pj, ord='fro') ** 2
    add = 0
    W = np.zeros((0, K))
    length = np.zeros(K)  # path lengths for vectors in codebook
    if full_W:
        W_out = []
    nA = np.linalg.norm(A, ord='fro') ** 2

    while 10 * np.log10(nA / nmin) < SNRdBmin:
        D = dmp.power2decomp(np.copy(A).T, np.copy(P).T, 1).T @ P - A  # diff matrix
        if max_length_diff == np.Inf:
            A1b = dmp.power2decomp(np.copy(A).T, np.copy(P).T, 2).T  # combination possibilities
        else:
            A1b = dmp.power2decomp2d_lz_length(np.copy(A).T, np.copy(P).T, 2, np.flip(length), max_length_diff).T
        nmin = np.inf
        penmin = 1  # penalty factor
        nDq = np.linalg.norm(D, ord='fro') ** 2  # norm of D => error
        avg_length = 0
        for k in range(N):  # iterate through the output vectors
            # Assigning a penalty based on the codeword length
            idxs_sel = np.nonzero(np.flip(A1b[k, 0:int(N + add)]))  # extracting the selected indices
            if np.size(idxs_sel) == 1:
                length_cur = int(length[idxs_sel[0][0]] + 1)
            elif np.size(idxs_sel) == 2:
                length_cur = np.maximum(int(length[idxs_sel[0][0]] + 1), int(length[idxs_sel[0][1]]))
            else:
                raise ValueError(
                    'Too many coefficients found in A1b column in iteration add: ' + str(add) + ' k: ' + str(k))
            if length_penalty:  # disable the length based penalty based on the function input
                penalty = mu * np.abs(length_cur - avg_length)
            else:
                penalty = 1

            mse = nDq - np.linalg.norm(D[k, :]) ** 2 + np.linalg.norm(A1b[k, :] @ P - A[k, :]) ** 2  # joint error + swapped error for col k
            if penalty * mse < penmin * nmin:
                nmin = mse
                penmin = penalty
                kbest = k
        if np.shape(W)[0] == 0:
            W = np.expand_dims(A1b[kbest, :], axis=0)
        else:
            W = np.concatenate((W, np.zeros((np.shape(W)[0], 1))), axis=1)
            W = np.concatenate((W, np.expand_dims(A1b[kbest, :], axis=0)), axis=0)
        P = np.concatenate((np.expand_dims(W[add, :] @ P, axis=0), P), axis=0)
        if full_W:
            W_out.append(np.concatenate((np.expand_dims(A1b[kbest, :], axis=0), np.eye(np.shape(A1b)[1])), axis=0))
        add += 1
        if add >= max_add:  # break if maximum number of adds has been reached
            break
        # calculating the path length of the newly created codeword
        idxs_path = np.nonzero(np.flip(W[-1, :]))
        if np.size(idxs_path) == 1:
            length = np.append(length, int(length[idxs_path[0][0]] + 1))
        elif np.size(idxs_path) == 2:
            length = np.append(length, np.maximum(int(length[idxs_path[0][0]] + 1), int(length[idxs_path[0][1]] + 1)))
        else:
            raise ValueError('Too many coefficients found in W in iteration add: ' + str(add))

    Pj = dmp.power2decomp(np.copy(A).T, np.copy(P).T, 1).T
    P = Pj @ P
    if full_W:
        W = W_out
    return P, W, Pj, add
