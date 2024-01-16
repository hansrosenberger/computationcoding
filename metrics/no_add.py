import numpy as np


# Computes the number of additions required for W when an arbitrary vector x is multiplied to it from the right-hand
# side, i.e. returns add(Wx);
def no_add_MPA(W):
    N, K = W.shape
    no_add = 0
    for n in range(N):
        no_add_row = -1
        row_nz = np.abs(np.squeeze(W[n, np.nonzero(W[n, :])]))
        while not np.all((row_nz == 0)):
            no_add_row += row_nz.size
            min_p2 = row_nz - (2 * np.ones(row_nz.size)) ** np.floor(np.log2(row_nz))
            max_p2 = (2 * np.ones(row_nz.size)) ** np.ceil(np.log2(row_nz)) - row_nz
            row_nz = np.min(np.concatenate((np.expand_dims(min_p2, axis=0), np.expand_dims(max_p2, axis=0)), axis=0), axis=0)
            row_nz = row_nz[np.nonzero(row_nz)]
        if no_add_row > 0:
            no_add += no_add_row
    return no_add
