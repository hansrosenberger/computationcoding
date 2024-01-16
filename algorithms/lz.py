import numpy as np

import dmp


# Performs the Lempel-Ziv inspired decomposition/fully sequential decomposition of target matrix
# @args: A: target matrix to be decomposed; SNRdBmin: target SNR to be achieved by the decomposition; full_W: If false
# only outputs a matrix where each line computes/adds one node to the graph; diff_signs: If true prohibits only negative
# coefficients for computations (required for some implementations); max_add: If true terminates after max_add
# irrespective if SNRdBmin is reached or not
# @returns: P: approximation of the matrix by the decomposition performed; W: matrix factor(s) created by the
# decomposition; Pj: final projection matrix applied after the multiplication of all matrix factors; add: number of
# additions needed to reach the desired SNRdBmin (if add=max_add algorithm terminated earlier)
def fastlzdecomp(A, SNRdBmin, mu=1, full_W=True, diff_signs=True, max_add=np.Inf):
    N, K = A.shape
    P = np.eye(K, K)
    Pj = dmp.power2decomp(np.copy(A).T, np.copy(P).T, 1).T
    nmin = np.linalg.norm(A - Pj, ord='fro')**2
    f = 0
    add = 0
    W = np.zeros((0, K))
    if full_W:
        W_out = []
    nA = np.linalg.norm(A, ord='fro')**2

    while 10 * np.log10(nA/nmin) < SNRdBmin:
        D = dmp.power2decomp(np.copy(A).T, np.copy(P).T, 1).T @ P - A  # diff matrix
        A1b = dmp.power2decomp(np.copy(A).T, np.copy(P).T, 2, diff_signs=diff_signs).T  # combination possibilities
        nmin = np.inf
        penmin = 1  # penalty factor
        nDq = np.linalg.norm(D, ord='fro')**2  # norm of D => error
        for k in range(N):  # iterate through the output vectors
            if k > f:
                penalty = 1
            else:
                penalty = mu**2
            mse = nDq - np.linalg.norm(D[k, :])**2 + np.linalg.norm(A1b[k, :] @ P - A[k, :])**2  # joint error + swapped error for col k
            if penalty*mse < penmin*nmin:
                nmin = mse
                penmin = penalty
                kbest = k
        if kbest > f:
            f += 1
        else:
            f = 0
        if np.shape(W)[0] == 0:
            W = np.expand_dims(A1b[kbest, :], axis=0)
        else:
            W = np.concatenate((W, np.zeros((np.shape(W)[0], 1))), axis=1)
            W = np.concatenate((W, np.expand_dims(A1b[kbest, :], axis=0)), axis=0)
        P = np.concatenate((np.expand_dims(W[add, :] @ P, axis=0), P), axis=0)
        if full_W:
            W_out.append(np.concatenate((np.expand_dims(A1b[kbest, :], axis=0), np.eye(np.shape(A1b)[1])), axis=0))
        add += 1
        if add >= max_add:
            break

    Pj = dmp.power2decomp(np.copy(A).T, np.copy(P).T, 1).T
    P = Pj @ P
    if full_W:
        W = W_out
    return P, W, Pj, add
