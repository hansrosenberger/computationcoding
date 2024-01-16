import numpy as np
import time
import re
import os
from subprocess import check_output

from metrics import distortion


def mcm_decomp(mat, b=8, path_to_build=None):
    mat_quant, scale = quantmat_int(mat, b=b, signed=True)  # quantize matrix to integers (temporary rescaling)
    _, SQNR_lin = quantmat_inv(np.copy(mat_quant), scale, np.copy(mat))  # determine the SQNR of found approximation
    mat_str_list = parse_mat_mcmlist(mat_quant, zero_rem=True)  # convert quantized matrix to a list of strings, exclude zeros
    raw_code = getmcmdata(mat_str_list, path_to_build=path_to_build)  # run matrix columns through the ./acm build
    code_list = []
    for c in raw_code:
        code_list.append(extract_op(c))  # extract the code ops per col from strings
    map_idx_op_l, map_const_op_l, map_idx_shifts_l = [], [], []
    for c in code_list:
        m_idx_tmp, m_const_tmp, m_idx_shifts_tmp = mapping_gen(c)
        map_idx_op_l.append(m_idx_tmp)
        map_const_op_l.append(m_const_tmp)
        map_idx_shifts_l.append(m_idx_shifts_tmp)
    # generate depth/delay
    depth_delay = depth_buffer_calc(map_idx_op_l, map_const_op_l, map_idx_shifts_l)
    # add additional delay for tgt const depth differences
    # depth_delay_outadj = output_delay(map_idx_op_l, map_idx_shifts_l, depth_delay.copy(), mat.shape[0])
    add_tree = (mat.shape[1] - 1) * mat.shape[0]  # additions required for the adder tree
    add_l, neg_l, delay_l = calc_add_neg_delay(map_idx_op_l, map_idx_shifts_l, depth_delay, ele=True)
    return code_list, depth_delay, add_l, neg_l, delay_l, add_tree, SQNR_lin


def quantmat_int(mat, b=8, signed=True):
    val_max = np.ceil(np.log2(np.max(np.abs(mat))))
    if signed:
        scale = 2 ** (b - val_max - 1)
    else:
        scale = 2 ** (b - val_max)
    # TODO explicit handling for zero values
    mat_quant = (scale * mat).astype(np.int64)
    return mat_quant, scale


def quantmat_inv(mat_q_int, scale, mat=None):
    mat_q = (1 / scale) * mat_q_int.astype(np.float64)
    if mat is None:
        return mat_q
    else:
        return mat_q, distortion.SNRmat_lin(mat, mat_q)


def parse_mat_mcmlist(mat, zero_rem=False):
    array_list = mat.T.tolist()
    str_list = []
    for col in array_list:
        if zero_rem:
            # TODO add specific handling for zeros in the output set (here zeros are only removed, as they are free)
            col = [item for item in col if item != 0]
        str_list.append(' '.join(map(str, col)))
    return str_list


def extract_op(str_list):
    out_list = []
    for s in str_list:
        out_list.append(list(filter(lambda x: x != '', [x.strip() for x in re.split(' = |, |;|\(|\)|/\* |\*/', re.sub('t', '', s))])))
    return out_list


def mapping_gen(str_list):
    map_idx_op = [(0, 0, 0, 1)]
    map_const_op = [1]
    map_idx_shifts = [[]]
    for s in str_list:
        if s[1] == 'add' or s[1] == 'sub':
            if s[1] == 'add':
                map_idx_op.append((int(s[0]), int(s[2]), int(s[3]), 1))
            elif s[1] == 'sub':
                map_idx_op.append((int(s[0]), int(s[2]), int(s[3]), -1))
            else:
                raise ValueError('operation not supported: ' + str(s[1]))
            map_const_op.append(int(s[4]))
            map_idx_shifts.append([])
        elif s[1] == 'shl' or s[1] == 'shr':
            root_idx = search_occur_idx(int(s[2]), map_idx_op, map_idx_shifts)
            if s[1] == 'shl':
                map_idx_shifts[root_idx].append((int(s[0]), int(s[3])))
            elif s[1] == 'shr':
                map_idx_shifts[root_idx].append((int(s[0]), -int(s[3])))
            else:
                raise ValueError('operation not supported: ' + str(s[1]))
        elif s[1] == 'neg':  # in case a variable is negated op => (x, y, -1, -1) (x neg var, y orig var)
            map_idx_op.append((int(s[0]), int(s[2]), -1, -1))
            map_const_op.append(int(s[3]))
            map_idx_shifts.append([])
        elif len(s) == 3:  # catch case that one constant equals another
            root_idx = search_occur_idx(int(s[1]), map_idx_op, map_idx_shifts)
            # tmp_idx = map_idx_op.index([item for item in map_idx_op if item[0] == int(s[1])][0])
            map_idx_shifts[root_idx].append((int(s[0]), 0))
        else:
            raise ValueError('operation not supported')
    return map_idx_op, map_const_op, map_idx_shifts


def getmcmdata(str_list, path_to_build=None, debug=True):
    transfer_file = 'py_transfer'
    poll_interval = 1
    raw_in = []
    if path_to_build is None and os.path.isfile(transfer_file):
        os.system('rm ' + transfer_file)
    elif os.path.isfile(path_to_build + '/' + transfer_file):
        os.system('rm ' + path_to_build + '/' + transfer_file)
    for col in str_list:
        init_time = time.time()
        if path_to_build == '':
            cmd_string = './acm ' + col + ' -code -noheader'
        else:
            cmd_string = path_to_build + '/acm ' + col + ' -code -noheader'
        out = check_output(cmd_string, shell=True).splitlines()
        raw_in.append([x.decode('utf-8') for x in out])
    return raw_in  # acm generated code (each sublist contains the code for the corresponding constant set (matrix column))


def depth_buffer_calc(m_idx_op_l, m_const_op_l, m_idx_shifts_l):
    if len(m_idx_op_l) != len(m_const_op_l) != len(m_idx_shifts_l):
        raise ValueError('input list lengths do not match')
    depth_buffer = []  # list contains a list of depth buffer tuples for each column of the original quantized matrix
    for col in range(len(m_idx_op_l)):  # iterate through different cols
        depth_buffer_tmp = []  # two ele tuples (depth, buffer)
        for op in m_idx_op_l[col]:
            if op == (0, 0, 0, 1):  # root node
                depth_buffer_tmp.append((0, 0))
            elif op[2] == -1:  # inverter op (no add/sub)
                root_idx_inv = search_occur_idx(op[1], m_idx_op_l[col], m_idx_shifts_l[col])
                depth_buffer_tmp.append((depth_buffer_tmp[root_idx_inv][0] + 1, 0))
            else:  # all other/following than root node
                # find both preceding/roots nodes
                # first operand
                root_idx_0 = search_occur_idx(op[1], m_idx_op_l[col], m_idx_shifts_l[col])
                # second operand
                root_idx_1 = search_occur_idx(op[2], m_idx_op_l[col], m_idx_shifts_l[col])
                # Find the current depth from both preceding/root nodes
                cur_depth = np.maximum(depth_buffer_tmp[root_idx_0][0], depth_buffer_tmp[root_idx_1][0]) + 1
                # add new node to depth_buffer list
                depth_buffer_tmp.append((cur_depth, 0))
                # update the delay value of both preceding/root nodes
                depth_buffer_tmp[root_idx_0] = (depth_buffer_tmp[root_idx_0][0], np.maximum(0, cur_depth - depth_buffer_tmp[root_idx_0][0] - 1))
                depth_buffer_tmp[root_idx_1] = (depth_buffer_tmp[root_idx_1][0], np.maximum(0, cur_depth - depth_buffer_tmp[root_idx_1][0] - 1))
        depth_buffer.append(depth_buffer_tmp)
    return depth_buffer


def search_occur_idx(op, m_idx_op, m_idx_shifts):
    root_op = [item for item in m_idx_op if item[0] == int(op)]
    if len(root_op) == 0:  # shifted version of op
        root_op_shifts = [item_l for item in m_idx_shifts for item_l in item if item_l[0] == int(op)]
        if len(root_op_shifts) == 1:  # catch case for multiple indices
            root_idx = nestl_idx(m_idx_shifts, root_op_shifts[0])[0]
        else:
            raise ValueError('multiple idxs detected')
    elif len(root_op) == 1:  # catch case for multiple indices
        root_idx = m_idx_op.index(root_op[0])
    else:
        raise ValueError('multiple idxs detected')
    return root_idx


def nestl_idx(nestl, obj):  # find all idx tuples of obj in nested list nestl
    occur = []
    for subl in nestl:
        if obj in subl:
            occur.append((nestl.index(subl), subl.index(obj)))
    if len(occur) == 0:
        return None
    elif len(occur) == 1:
        return occur[0]
    else:
        return occur


def output_delay(m_idx_op_l, m_idx_shifts_l, depth_delay_l, rows):
    if len(m_idx_op_l) != len(m_idx_shifts_l) != len(depth_delay_l):
        raise ValueError('m_idx_op, delay_depth, m_idx_shifts do not match')
    for col in range(len(m_idx_op_l)):
        out_root_idx = []
        max_depth = 0
        for out_idx in np.arange(1, rows + 1, 1):  # find the root_idxs for all tgt constants and find the maximum depth
            root_idx = search_occur_idx(out_idx, m_idx_op_l[col], m_idx_shifts_l[col])
            out_root_idx.append(root_idx)
            max_depth = np.maximum(max_depth, depth_delay_l[col][root_idx][0])
        for root_idx in out_root_idx:
            depth_delay_l[col][root_idx] = (depth_delay_l[col][root_idx][0], np.maximum(0, max_depth - depth_delay_l[col][root_idx][0] - 1))
    return depth_delay_l


def calc_add_neg_delay(m_idx_op_l, m_idx_shifts_l, depth_delay_l, ele=False):
    if len(m_idx_op_l) != len(m_idx_shifts_l) != len(depth_delay_l):
        raise ValueError('m_idx_op, delay_depth, m_idx_shifts do not match')
    add_col, neg_col, delay_col = [], [], []
    for col in range(len(m_idx_op_l)):
        add_tmp = len([item for item in m_idx_op_l[col] if item[2] != -1])
        neg_col.append(len(m_idx_op_l[col]) - add_tmp)
        delay_tmp = 0
        for tup in depth_delay_l[col]:
            delay_tmp += tup[1]
        add_col.append(add_tmp)
        delay_col.append(delay_tmp)
    if ele:
        return add_col, neg_col, delay_col
    else:
        return sum(add_col), sum(neg_col), sum(delay_col)


def main():
    A = np.random.normal(0, 1, (64, 4))
    code_list, depth_delay, _, _, _, _, SQNR = mcm_decomp(A, b=20, path_to_build='Insert your path to the C++ MCM build here')


if __name__ == "__main__":
    main()

