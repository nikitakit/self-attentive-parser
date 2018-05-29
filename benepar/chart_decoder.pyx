import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython

ctypedef np.float32_t DTYPE_t

# @cython.boundscheck(False)
def decode(np.ndarray[DTYPE_t, ndim=3] label_scores_chart):
    cdef DTYPE_t NEG_INF = -np.inf
    cdef int sentence_len = label_scores_chart.shape[0] - 1

    cdef np.ndarray[DTYPE_t, ndim=2] value_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.float32)
    cdef np.ndarray[int, ndim=2] split_idx_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.int32)
    cdef np.ndarray[int, ndim=2] best_label_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.int32)

    cdef int length
    cdef int left
    cdef int right

    cdef DTYPE_t label_score
    cdef int argmax_label_index

    cdef int best_split
    cdef int split_idx # Loop variable for splitting
    cdef DTYPE_t split_val # best so far
    cdef DTYPE_t max_split_val

    cdef int label_index_iter

    for length in range(1, sentence_len + 1):
        for left in range(0, sentence_len + 1 - length):
            right = left + length

            # We do argmax ourselves to make sure it compiles to pure C
            if length < sentence_len:
                argmax_label_index = 0
            else:
                # Not-a-span label is not allowed at the root of the tree
                argmax_label_index = 1

            label_score = label_scores_chart[left, right, argmax_label_index]
            for label_index_iter in range(1, label_scores_chart.shape[2]):
                if label_scores_chart[left, right, label_index_iter] > label_score:
                    argmax_label_index = label_index_iter
                    label_score = label_scores_chart[left, right, label_index_iter]
            best_label_chart[left, right] = argmax_label_index

            if length == 1:
                value_chart[left, right] = label_score
                continue

            best_split = left + 1
            split_val = NEG_INF
            for split_idx in range(left + 1, right):
                max_split_val = value_chart[left, split_idx] + value_chart[split_idx, right]
                if max_split_val > split_val:
                    split_val = max_split_val
                    best_split = split_idx

            value_chart[left, right] = label_score + value_chart[left, best_split] + value_chart[best_split, right]
            split_idx_chart[left, right] = best_split

    # Now we need to recover the tree by traversing the chart starting at the
    # root. This iterative implementation is faster than any of my attempts to
    # use helper functions and recursion

    # All fully binarized trees have the same number of nodes
    cdef int num_tree_nodes = 2 * sentence_len - 1
    cdef np.ndarray[int, ndim=1] included_i = np.empty(num_tree_nodes, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] included_j = np.empty(num_tree_nodes, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] included_label = np.empty(num_tree_nodes, dtype=np.int32)

    cdef int idx = 0
    cdef int stack_idx = 1
    # technically, the maximum stack depth is smaller than this
    cdef np.ndarray[int, ndim=1] stack_i = np.empty(num_tree_nodes + 5, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] stack_j = np.empty(num_tree_nodes + 5, dtype=np.int32)
    stack_i[1] = 0
    stack_j[1] = sentence_len

    cdef int i, j, k
    while stack_idx > 0:
        i = stack_i[stack_idx]
        j = stack_j[stack_idx]
        stack_idx -= 1
        included_i[idx] = i
        included_j[idx] = j
        included_label[idx] = best_label_chart[i, j]
        idx += 1
        if i + 1 < j:
            k = split_idx_chart[i, j]
            stack_idx += 1
            stack_i[stack_idx] = k
            stack_j[stack_idx] = j
            stack_idx += 1
            stack_i[stack_idx] = i
            stack_j[stack_idx] = k

    cdef DTYPE_t score = value_chart[0, sentence_len]

    return score, included_i.astype(int), included_j.astype(int), included_label.astype(int)
