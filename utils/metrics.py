import numpy as np

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Evaluation protocol for Market-1501 dataset.
    Same identity from the same camera is excluded from gallery search.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Warning: gallery size {num_g} is smaller than max_rank, setting max_rank={max_rank}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc and mAP
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition arises when query identity does not exist in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not exist in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(axis=0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def compute_distance_matrix(query_feats, gallery_feats, metric='euclidean'):
    """
    Computes distance matrix between query features and gallery features.
    query_feats: (N, D)
    gallery_feats: (M, D)
    """
    if metric == 'cosine':
        # Assumes normalized features
        distmat = 1.0 - np.dot(query_feats, gallery_feats.T)
    else:  # Euclidean distance
        m, n = query_feats.shape[0], gallery_feats.shape[0]
        distmat = np.power(query_feats, 2).sum(axis=1, keepdims=True) + \
                  np.power(gallery_feats, 2).sum(axis=1, keepdims=True).T
        distmat = distmat - 2 * np.dot(query_feats, gallery_feats.T)
        distmat = np.maximum(distmat, 0.0)
        distmat = np.sqrt(distmat)
    return distmat
