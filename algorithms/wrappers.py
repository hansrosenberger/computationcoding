# Wrapper functions for decomposition algorithms for parallelization with ray
import lz
from lz_vis import delayregisters_no
import rs
import dmp
import graph
from parallel_graph import decomp_pwr2_graph
from mcm import mcm_decomp
from metrics import distortion as dist
from metrics.no_add import no_add_MPA

import numpy as np
import ray


@ray.remote
def decomp_MPA(target_mat, codebook_mat_init, S_MPA, no_itr):
    K = target_mat.shape[0]
    W_MPA = np.zeros((K, K, no_itr))
    P_MPA = np.zeros(target_mat.shape)
    no_add_MPA = np.zeros(no_itr)
    SQNR_MPA_lin = np.zeros(no_itr)
    for itr in range(no_itr):
        print('Iteration: ' + str(itr))
        if itr == 0:
            W_MPA[:, :, itr] = dmp.power2decomp(np.copy(target_mat).T, np.copy(codebook_mat_init).T, S_MPA).T
            P_MPA = W_MPA[:, :, itr] @ codebook_mat_init
            no_add_MPA[itr] = (S_MPA - 1) * K
        else:
            W_MPA[:, :, itr] = dmp.power2decomp(np.copy(target_mat).T, np.copy(P_MPA).T, S_MPA).T
            P_MPA = W_MPA[:, :, itr] @ P_MPA
            no_add_MPA[itr] = no_add_MPA[int(itr - 1)] + (S_MPA - 1) * K
        SQNR_MPA_lin[itr] = dist.SNRmat_lin(np.copy(target_mat), np.copy(P_MPA))
    return P_MPA, W_MPA, no_add_MPA, SQNR_MPA_lin


@ray.remote
def decomp_MPA_metrics(target_mat, codebook_mat_init, S, no_mat, SQNR_tgt=None, transpose=True, diff_signs=False,
                       wiring_red=True, pos_signs=False):
    return dmp.decomp_pwr2_metrics(target_mat, codebook_mat_init, S, no_mat, SQNR_tgt=SQNR_tgt, transpose=transpose,
                               diff_signs=diff_signs, wiring_red=wiring_red, pos_signs=pos_signs)


@ray.remote
def decomp_expo_reduced(target_mat, codebook_mat_init, S_red, mem, no_itr, MPA_init=0, S_MPA=2):
    K = target_mat.shape[0]
    W_red = np.zeros((K, K, no_itr))
    P_red = np.zeros(target_mat.shape)
    no_add_red = np.zeros(no_itr)
    SQNR_red_lin = np.zeros(no_itr)
    for itr in range(no_itr):
        if itr == 0:
            if MPA_init > 0:
                W_red[:, :, itr] = dmp.power2decomp(np.copy(target_mat).T, np.copy(codebook_mat_init).T, S_MPA).T
                P_red = W_red[:, :, itr] @ codebook_mat_init
                no_add_red[itr] = (S_MPA - 1) * K
            else:
                W_red[:, :, itr] = rs.decomp_pwr2_expo_reduced(np.copy(target_mat).T, np.copy(codebook_mat_init).T,
                                                               S_red, mem).T
                P_red = W_red[:, :, itr] @ codebook_mat_init
                no_add_red[itr] = (S_red - 1) * K
        else:
            if itr < MPA_init:
                W_red[:, :, itr] = dmp.power2decomp(np.copy(target_mat).T, np.copy(P_red).T, S_MPA).T
                P_red = W_red[:, :, itr] @ P_red
                no_add_red[itr] = no_add_red[int(itr - 1)] + (S_MPA - 1) * K
            else:
                W_red[:, :, itr] = rs.decomp_pwr2_expo_reduced(np.copy(target_mat).T, np.copy(P_red).T, S_red, mem).T
                P_red = W_red[:, :, itr] @ P_red
                no_add_red[itr] = no_add_red[int(itr - 1)] + (S_red - 1) * K
        SQNR_red_lin[itr] = dist.SNRmat_lin(np.copy(target_mat), np.copy(P_red))
    return P_red, W_red, no_add_red, SQNR_red_lin


@ray.remote
def decomp_fastlz(A, SNRdBmin, mu=1, diff_signs=True, max_add=np.Inf, add_step=10, delay_cost=False):
    P, W, _, add_ttl = lz.fastlzdecomp(A, SNRdBmin, mu=mu, full_W=False, diff_signs=diff_signs,
                                                        max_add=max_add)
    C = np.eye(A.shape[1])
    SQNR_lin_list = []
    add_list = []
    delay_list = []
    for add in range(W.shape[0]):
        C = np.concatenate((np.expand_dims(W[add, 0:int(A.shape[1] + add)] @ C, axis=0), C), axis=0)
        if np.mod(add + 1, add_step) == 0:
            add_list.append(add + 1)
            if delay_cost:
                delay_list.append(delayregisters_no(np.copy(W[0:add, 0:(add + A.shape[1])]), np.eye(A.shape[1])))
            SQNR_lin_list.append(dist.SNRmat_lin(A, dmp.power2decomp(np.copy(A).T, np.copy(C).T, 1).T @ C))
    if delay_cost:
        return W, np.array(add_list), np.array(SQNR_lin_list), np.array(delay_list)
    else:
        return W, np.array(add_list), np.array(SQNR_lin_list)


@ray.remote
def decomp_fastlzpath(A, SNRdBmin, mu=1, max_length_diff=np.Inf, full_W=False, length_penalty=True, max_add=np.Inf,
                      add_step=10, delay_cost=False):
    P, W, _, add_ttl = graph.fastlzdecomp_pathopt(A, SNRdBmin, mu=mu, max_length_diff=max_length_diff,
                                                                full_W=full_W, length_penalty=length_penalty,
                                                                max_add=max_add)
    C = np.eye(A.shape[1])
    SQNR_lin_list = []
    add_list = []
    delay_list = []
    for add in range(W.shape[0]):
        C = np.concatenate((np.expand_dims(W[add, 0:int(A.shape[1] + add)] @ C, axis=0), C), axis=0)
        if np.mod(add + 1, add_step) == 0:
            add_list.append(add + 1)
            if delay_cost:
                delay_list.append(delayregisters_no(np.copy(W[0:add, 0:(add + A.shape[1])]), np.eye(A.shape[1])))
            SQNR_lin_list.append(dist.SNRmat_lin(A, dmp.power2decomp(np.copy(A).T, np.copy(C).T, 1).T @ C))
    if delay_cost:
        return W, np.array(add_list), np.array(SQNR_lin_list), np.array(delay_list)
    else:
        return W, np.array(add_list), np.array(SQNR_lin_list)


@ray.remote
def decomp_parallelgraph(target_mat, SQNR=47, mem=1, S_parallel=3):
    _, W, _, _ = decomp_pwr2_graph(target_mat, SQNR_dB=SQNR, mem=mem, S_parallel=S_parallel)
    C_init = np.eye(target_mat.shape[1])
    no_add = []
    SQNR = []
    for W_mat in W:
        C_init = W_mat @ C_init
        SQNR.append(dist.SNRmat_lin(target_mat, dmp.power2decomp(np.copy(target_mat).T, np.copy(C_init).T,
                                                                         1).T @ C_init))
        if len(no_add) == 0:
            no_add.append(no_add_MPA(np.copy(W_mat)))
        else:
            no_add.append(no_add[-1] + no_add_MPA(np.copy(W_mat)))
    return W, np.array(no_add), np.array(SQNR)


@ray.remote
def decomp_mcm(mat, bws=np.arange(4, 20, 2), path_to_build=None):
    adds, adds_tree, negs, dels, SQNR_lin = [], [], [], [], []
    for b in bws:
        _, _, add_l_tmp, neg_l_tmp, delay_l_tmp, add_tree_tmp, SQNR_lin_tmp = mcm_decomp(np.copy(mat), b=b,
                                                                                     path_to_build=path_to_build)
        adds.append(sum(add_l_tmp))
        adds_tree.append(add_tree_tmp)
        negs.append(sum(neg_l_tmp))
        dels.append(sum(delay_l_tmp))
        SQNR_lin.append(SQNR_lin_tmp)
    return adds, adds_tree, negs, dels, SQNR_lin
