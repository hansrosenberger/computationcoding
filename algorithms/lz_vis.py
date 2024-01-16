import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout


# Visualizes a decomposition of the LZ inspired algorithm as a di-graph using networkx and graphviz (both packages are
# required)
# @args: W, C, Pj: the output of a LZ decomposition performed by the function lzdecomp; addmax: the maximum number of
# additions set for the decomposition
# @returns: G: the networkx graph object; pos: the positions (as a dictionary) of the nodes and edges produced by
# graphviz using the dot layout; codewords: the codewords corresponding to the nodes of the graph, indices of the graph
# nodes correspond to the indices of the codewords in the dictionary
# @notes: Produces a plot of the graph internally and displays it in a matplotlib.pyplot plot
def vis_lz_graph(A, W, C, Pj, addmax, title='', full_W=True):
    codewords = []
    G = nx.DiGraph()
    K, N = A.shape

    if not full_W:
        addmax = np.shape(W)[0]

    # no_initcodewords = np.minimum(np.shape(C)[0], np.shape(C)[1])
    # C_0_shape = np.shape(C)[0]
    no_initcodewords = np.shape(C)[1]
    for i in range(no_initcodewords):
        G.add_node(i, label=str(i))
        codewords.append(np.flipud(C)[i, :])

    for i in range(addmax):
        if full_W:
            cur_w = np.flip(W[0, 0:int(no_initcodewords + i), i])
        else:
            cur_w = np.flip(W[i, 0:int(no_initcodewords + i)])
        G.add_node(int(no_initcodewords + i), label=str(no_initcodewords + i))
        if np.count_nonzero(cur_w) == 1:
            idxs = np.nonzero(cur_w)
            # ld_idx = np.log2(cur_w[idxs[0][0]])
            G.add_edge(idxs[0][0], int(no_initcodewords + i))
            # if np.mod(ld_idx, 1) < 0.5:
            # TBD
            # else:
            # TBD
        elif np.count_nonzero(cur_w) == 2:
            idxs = np.nonzero(cur_w)
            G.add_edge(idxs[0][0], int(no_initcodewords + i))
            G.add_edge(idxs[0][1], int(no_initcodewords + i))
        else:
            raise ValueError('Error in graph construction')
        if full_W:
            C = W[:, :, i] @ C  # TODO fix for full_W=False
        else:
            C = np.concatenate((np.expand_dims(W[i, 0:int(no_initcodewords + i)] @ C, axis=0), C), axis=0)
        codewords.append(C[0, :])

    write_dot(G, 'test.dot')

    if not title:
        plt.title('Decomposition Graph (' + str(K) + 'x' + str(N) + ', ' + str(addmax) + ' additions)')
    else:
        plt.title(title)
    pos = graphviz_layout(G, prog="dot")
    # drawing the graph
    nx.draw(G, pos, with_labels=False, arrows=True)
    nx.draw_networkx_nodes(G, pos, nodelist=np.arange(no_initcodewords), node_color='tab:green')
    if N - K > 0:
        tmp = np.nonzero(np.fliplr(Pj))[1]
        nx.draw(G, pos, nodelist=(tmp-((N-K)*np.ones(tmp.shape))), node_color='tab:red')
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=np.nonzero(np.fliplr(Pj))[1], node_color='tab:red')
    labels = {}
    for i in range(int(np.shape(C)[1] + addmax)):
        labels[i] = str(i)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')
    # plt.savefig('nx_test.png')
    plt.show()
    return G, pos, codewords


def cw_length(W, C):  # Calculates the path length of each codeword within the codebook
    no_cw = np.shape(W)[0]
    no_init_cw = np.shape(C)[1]
    length = np.zeros(no_init_cw)
    for idx in range(no_cw):
        idxs_path = np.nonzero(np.flip(W[idx, 0:int(no_init_cw + idx)]))
        if np.size(idxs_path) == 1:
            length = np.append(length, int(length[idxs_path[0][0]] + 1))
        elif np.size(idxs_path) == 2:
            length = np.append(length, np.maximum(int(length[idxs_path[0][0]] + 1), int(length[idxs_path[0][1]] + 1)))
    return length


# Returns the delay register count given a wiring (LZ/MA style wiring matrix) and a codebook matrix
def delayregisters_no(W, C, separate=False):
    no_cw = np.shape(W)[0]
    no_init_cw = np.shape(C)[1]
    length = cw_length(np.copy(W), np.copy(C))
    maxdepthused = np.zeros(np.shape(length))
    for idx in range(no_cw):
        idxs_path = np.nonzero(np.flip(W[idx, 0:int(no_init_cw + idx)]))
        if np.size(idxs_path) == 1:
            maxdepthused[idxs_path[0][0]] = np.maximum(maxdepthused[idxs_path[0][0]], length[int(no_init_cw + idx)] - length[idxs_path[0][0]])
        elif np.size(idxs_path) == 2:
            maxdepthused[idxs_path[0][0]] = np.maximum(maxdepthused[idxs_path[0][0]], length[int(no_init_cw + idx)] - length[idxs_path[0][0]])
            maxdepthused[idxs_path[0][1]] = np.maximum(maxdepthused[idxs_path[0][1]], length[int(no_init_cw + idx)] - length[idxs_path[0][1]])
    if separate:
        maxdepthused[maxdepthused == 0] = 1
        return maxdepthused - np.ones(np.shape(maxdepthused))
    else:
        maxdepthused[maxdepthused == 0] = 1
        return np.sum(maxdepthused - np.ones(np.shape(maxdepthused)))


# Determines the codebook of a certain depth given an LZ style input
def get_c_depth(W, C, depth):
    no_cw = np.shape(W)[0]
    no_init_cw = np.shape(C)[1]
    length = cw_length(np.copy(W), np.copy(C))
    for idx in range(no_cw):
        C = np.concatenate((np.expand_dims(W[idx, 0:int(no_init_cw + idx)] @ C, axis=0), C), axis=0)
    return C[np.flip(length) == depth]

