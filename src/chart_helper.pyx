import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython

ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
def decode(int force_gold, int sentence_len, np.ndarray[DTYPE_t, ndim=3] label_scores_chart, int is_train, gold, label_vocab):
    cdef DTYPE_t NEG_INF = -np.inf

    cdef np.ndarray[DTYPE_t, ndim=2] value_chart = np.zeros((sentence_len+2, sentence_len+2), dtype=np.float32)
    cdef np.ndarray[int, ndim=2] split_idx_chart = np.zeros((sentence_len+2, sentence_len+2), dtype=np.int32)
    cdef np.ndarray[int, ndim=2] best_label_chart = np.zeros((sentence_len+2, sentence_len+2), dtype=np.int32)

    cdef int length
    cdef int left
    cdef int right

    cdef np.ndarray[DTYPE_t, ndim=1] label_scores_for_span

    cdef int oracle_label_index
    cdef DTYPE_t label_score
    cdef int argmax_label_index
    cdef DTYPE_t left_score
    cdef DTYPE_t right_score

    cdef int best_split
    cdef int split_idx # Loop variable for splitting
    cdef DTYPE_t split_val # best so far
    cdef DTYPE_t max_split_val

    for length in range(1, sentence_len + 1):
        for left in range(0, sentence_len + 1 - length):
            right = left + length

            label_scores_for_span = label_scores_chart[left, right]

            if is_train:
                oracle_label_index = label_vocab.index(gold.oracle_label(left, right))

            if force_gold:
                label_score = label_scores_for_span[oracle_label_index]
                best_label_chart[left, right] = oracle_label_index

            else:
                if is_train:
                    # augment
                    label_scores_for_span = label_scores_for_span + 1
                    label_scores_for_span[oracle_label_index] -= 1
                if length < sentence_len:
                    argmax_label_index = label_scores_for_span.argmax()
                else:
                    argmax_label_index = label_scores_for_span[1:].argmax() + 1
                label_score = label_scores_for_span[argmax_label_index]
                best_label_chart[left, right] = argmax_label_index

            if length == 1:
                value_chart[left, right] = label_score
                continue

            if force_gold:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                best_split = oracle_split
            else:
                best_split = left + 1
                split_val = NEG_INF
                for split_idx in range(left + 1, right):
                    max_split_val = value_chart[left, split_idx] + value_chart[split_idx, right]
                    if max_split_val > split_val:
                        split_val = max_split_val
                        best_split = split_idx

            value_chart[left, right] = label_score + value_chart[left, best_split] + value_chart[best_split, right]
            split_idx_chart[left, right] = best_split

    # TODO(nikita): optimize the back-pointer computations
    included_i = []
    included_j = []
    included_label = []

    split_idx_chart_ = split_idx_chart[:,:]
    value_chart_ = value_chart[:,:]
    best_label_chart_ = best_label_chart[:,:]
    running_val = [0.0]
    def inner(i, j):
        included_i.append(i)
        included_j.append(j)
        included_label.append(best_label_chart_[i, j])
        if i + 1 < j:
            k = split_idx_chart_[i, j]
            inner(i, k)
            inner(k, j)
    inner(0, sentence_len)

    running_total = 0.0
    for idx in range(0, len(included_i)):
        running_total += label_scores_chart[included_i[idx], included_j[idx], included_label[idx]]

    score = value_chart[0, sentence_len]
    augment_amount = round(score - running_total)
    included_i = np.array(included_i, dtype=int)
    included_j = np.array(included_j, dtype=int)
    included_label = np.array(included_label, dtype=int)

    return score, included_i, included_j, included_label, augment_amount
