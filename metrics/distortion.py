import numpy as np


# Determine the S(Q)NR of an approximated matrix relative to the target (error calculated element-wise)
# @args: target_mat: target matrix, app_mat: approximated matrix
# @returns: out: SNR scalar
# @note: Both inputs need to be in floating point matrix representations
def SNRmat(target_mat, app_mat):
    if target_mat.ndim > 2 or app_mat.ndim > 2:
        raise ValueError('Wrong Array Dimensions on the input variables')
    out = 20 * np.log10(np.linalg.norm(target_mat)/np.linalg.norm(target_mat - app_mat))
    return out


# Same function as above; however returns the linear S(Q)NR
def SNRmat_lin(target_mat, app_mat):
    if target_mat.ndim > 2 or app_mat.ndim > 2:
        raise ValueError('Wrong Array Dimensions on the input variables')
    out = np.linalg.norm(target_mat)/np.linalg.norm(target_mat - app_mat)
    return out
