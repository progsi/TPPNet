import numpy as np
import time
from multiprocessing import Pool
import os, random
import torch
import torch.utils
# other
def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def calc_MAP(array2d, version, que_range=None, K=1e10):
    if que_range is not None:
        que_s, que_t = que_range[0], que_range[1]
        if que_s == 0:
            ref_s, ref_t = que_t, len(array2d)
        else:
            ref_s, ref_t = 0, que_s
    else:
        que_s, que_t, ref_s, ref_t = 0, len(array2d), 0, len(array2d)

    new_array2d = []
    for u, row in enumerate(array2d[que_s: que_t]):
        row = [(v + ref_s, col) for (v, col) in enumerate(row[ref_s: ref_t]) if u + que_s != v + ref_s]
        new_array2d.append(row)
    MAP, top10, rank1 = 0, 0, 0
    
    for u, row in enumerate(new_array2d):
        row.sort(key=lambda x: x[1])
        per_top10, per_rank1, per_MAP = 0, 0, 0
        version_cnt = 0.
        u = u + que_s
        for k, (v, val) in enumerate(row):
            
            if version[u] == version[v]:
                
                if k < K:
                    version_cnt += 1
                    per_MAP += version_cnt / (k + 1)
                if per_rank1 == 0:
                    per_rank1 = k + 1
                if k < 10:
                    per_top10 += 1
        per_MAP /= 1 if version_cnt < 0.0001 else version_cnt
        # if per_MAP < 0.1:
        #     print row
        MAP += per_MAP
        top10 += per_top10
        rank1 += per_rank1
    return MAP / float(que_t - que_s), top10 / float(que_t - que_s) / 10, rank1 / float(que_t - que_s)


def compute_map(X, labels):
    
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import average_precision_score
    # Convert labels to binary matrix
    lb = LabelBinarizer()
    Y_true = lb.fit_transform(labels)

    # Compute pairwise similarity matrix
    sim_mat = X / np.max(X, axis=1, keepdims=True)

    # Compute AP for each class
    ap_scores = []
    for c in range(Y_true.shape[1]):
        ap = average_precision_score(Y_true[:, c], sim_mat[:, c])
        ap_scores.append(ap)

    # Compute MAP
    map_score = np.mean(ap_scores)
    return map_score


