import numpy as np

from lz_vis import cw_length


# Given a wiring matrix, this function removes unused computations/vertices in the graph with indegree 0 (except output
# vertices)
def wiring_red(W, list=True):
    if list:
        for w in range(int(len(W) - 1), 0, -1):  # The wiring matrix with index 0 does not need to be checked
            N, K = W[w].shape
            W_tmp = np.copy(W[w])
            for i in range(K):
                if np.all(W_tmp[:, i] == 0):
                    W[int(w - 1)][i, :] = np.zeros(K)
    else:
        for w in range(int(W.shape[2] - 1), 0, -1):
            W_tmp = np.copy(W[:, :, w])
            N, K = W_tmp.shape
            for i in range(K):
                if np.all(W_tmp[:, i] == 0):
                    W[i, :, int(w - 1)] = np.zeros(K)
    return W


def wiringred_LZparallel(W, max_depth=np.Inf):
    if W.ndim != 2:
        raise ValueError('W ndim invalid: ' + str(W.ndim))
    add, tmp = W.shape
    input_dim = tmp - add + 1
    w_list = wiring_conv(np.copy(W))
    W_out = []
    length = cw_length(np.copy(W), np.eye(input_dim))
    layers, cw_count = np.unique(length, return_counts=True)
    input_tup = (0, 0)
    for l in layers:
        if l == 0:  # only inputs
            input_tup = (0, input_dim)
        elif l == max_depth + 1:
            break
        else:  # subsequent layers
            idx_array = np.where(length == l)[0] - input_dim
            W_out.append(w_stripmerge(w_list, idx_array, input_tup))
            input_tup = (input_tup[1], input_tup[1] + idx_array.size)
    return W_out


# converts a dense LZ wiring matrix to a list of wiring vectors shortened to the current codebook length
def wiring_conv(W):
    if W.ndim != 2:
        raise ValueError('W ndim invalid: ' + str(W.ndim))
    add, tmp = W.shape
    input_dim = tmp - add + 1
    w_list = []
    for a in range(add):
        # w_list.append(W[a, 0:(input_dim + a)])
        w_list.append(np.flip(W[a, 0:(input_dim + a)]))
    return w_list  # CAUTION: w_list vectors are flipped, add unflip if not stripmerge is used directly after


def w_stripmerge(w_list, idx_array, input_dim):
    w_list_short = []
    for i in idx_array:  # remove w vectors from undesired layers and purge unnecessary zeros
        w_list_short.append(np.expand_dims(np.flip(w_list[i][input_dim[0]:input_dim[1]]), axis=0))
    w_list_short.reverse()
    return np.concatenate(w_list_short)
