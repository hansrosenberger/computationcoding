import numpy as np

from graph import fastlzdecomp_pathopt
from lz_vis import cw_length, get_c_depth
from wiring_red import wiringred_LZparallel
from dmp import power2decomp
from rs import decomp_pwr2_expo_reduced
from metrics.distortion import SNRmat


# Combination of the mixed algorithm constrained to a fully parallel structure for codebook build up and the reduced
# state algorithm for codebook refinement. Finds the best tradeoff between the two by a full search
def decomp_pwr2_graph(target_mat, SQNR_dB=47, mem=1, S_parallel=3):
    # matrix dimensions
    if target_mat.ndim != 2:
        raise ValueError('target_mat ndim invalid: ' + str(target_mat.ndim))
    N, K = target_mat.shape
    # create the initial codebooks with the path constrained mixed algorithm
    _, W, Pj, add = fastlzdecomp_pathopt(np.copy(target_mat), SNRdBmin=SQNR_dB, mu=1, max_length_diff=0, full_W=False,
                                         length_penalty=True, max_add=np.Inf)
    # return the number of codewords in each layer
    length = cw_length(np.copy(W), np.eye(K))
    layers, cw_count = np.unique(length, return_counts=True)

    codebook_lz_layer = []
    # calculate the codebooks for each layer until N elements are reached
    for l in range(int(np.max(length) - 1)):
        codebook_lz_layer.append(get_c_depth(np.copy(W), np.eye(K), l + 1))
        if codebook_lz_layer[-1].shape[0] == N or l == int(np.max(length) - 1):
            max_depth_LZ = l  # max depth considered
            break

    try:
        max_depth_LZ
    except NameError:
        max_depth_LZ = int(np.max(length) - 1)

    # calculate for each of the codebooks (created by the mixed algorithm) determined before the further decomp by a
    # fully parallel algortihm (reduced state or fully parallel)
    codebooks = []
    wiringmats = []
    SQNR = []
    for i in range(len(codebook_lz_layer)):
        codebook_cur = codebook_lz_layer[0:int(i + 1)].copy()
        wiringmat_cur = []
        SQNR_cur = [SNRmat(target_mat, power2decomp(np.copy(target_mat).T, codebook_cur[-1].T, S=1, diff_signs=False,
                                                    pos_signs=False).T @ codebook_cur[-1])]
        while SQNR_cur[-1] < SQNR_dB:
            if mem == 1:  # fully parallel algorithm
                wiringmat_cur.append(power2decomp(np.copy(target_mat).T, codebook_cur[-1].T, S=S_parallel,
                                                  diff_signs=False, pos_signs=False).T)
            elif mem > 1:  # reduced-state algorithm
                wiringmat_cur.append(
                    decomp_pwr2_expo_reduced(np.copy(target_mat).T, codebook_cur[-1].T, S=S_parallel, mem=mem).T)
            else:
                raise ValueError('Memory parameter out of range: ' + str(mem))
            codebook_cur.append(wiringmat_cur[-1] @ codebook_cur[-1])
            SQNR_cur.append(SNRmat(target_mat, codebook_cur[-1]))
        wiringmats.append(wiringmat_cur)
        codebooks.append(codebook_cur)
        SQNR.append(SQNR_cur)

    # selection of the best mixed algorithm (path constrained) to fully parallel transition
    no_adds = []
    SQNR_add_ratio = []
    for idx in range(len(codebooks)):
        add_tmp = 0
        for c in codebooks[idx]:
            if c.shape[0] < N:
                add_tmp += c.shape[0]  # as always S_mixed = 2
            else:
                add_tmp += N * (S_parallel - 1)
        no_adds.append(add_tmp)
        SQNR_add_ratio.append(SQNR[idx][-1] / add_tmp)
    # idx_min = no_adds.index(min(no_adds))
    idx_min = SQNR_add_ratio.index(max(SQNR_add_ratio))  # select the index with the highest SQNR/add ratio (best slope)

    # creating the block wiring matrices from LZ and merging with the best result obtained with the fully
    # parallel algorithm
    W_LZ = wiringred_LZparallel(np.copy(W), max_depth=max_depth_LZ)
    W_out = W_LZ[0:(idx_min + 1)] + wiringmats[idx_min]
    return codebooks[idx_min], W_out, no_adds[idx_min], SQNR[idx_min][-1]


def main():  # Testing purposes
    N = 64
    K = 4
    A = np.random.normal(0, 1, (N, K))
    C, W, add, SQNR = decomp_pwr2_graph(np.copy(A), SQNR_dB=47, mem=50, S_parallel=3)
    return C, W, add, SQNR


if __name__ == "__main__":
    main()
