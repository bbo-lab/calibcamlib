import numpy as np
import scipy


def get_line_dist(r1, e1, r2, e2):
    n = np.cross(e1, e2)
    return np.abs(np.sum(n * (r1 - r2)))


def intersect(bases, vecs):
    p = np.empty(3)
    p[:] = np.NaN

    ray_ok = ~np.any([
        np.isnan(bases),
        np.isnan(vecs)
    ], axis=(0, 2))

    bases = bases[ray_ok]
    vecs = vecs[ray_ok]

    n = bases.shape[0]
    if n < 2:
        return p

    M = np.empty((n, 3, 3))
    Mbase = np.empty((n, 3, 1))

    for u in range(n):
        planebasis = scipy.linalg.null_space(vecs[np.newaxis, u])
        M[u] = planebasis @ planebasis.T
        Mbase[u] = M[u] @ bases[u, np.newaxis].T

    if np.linalg.matrix_rank(np.sum(M, axis=0)) < 3:
        return p

    return np.squeeze(np.linalg.solve(np.sum(M, axis=0), np.sum(Mbase, axis=0)).T)


def calc_3derr(X, P, V):
    dists = np.zeros(shape=(P.shape[0]))

    for i, p in enumerate(P):
        dists[i] = calc_min_line_point_dist(X, p, V[i])

    if np.all(np.isnan(dists)):
        return np.nansum(dists ** 2), dists
    else:
        return np.NaN, dists


def calc_min_line_point_dist(x, p, v):
    # print(x.shape)
    # print(p.shape)
    # print(v.shape)
    d = x - p
    dist = np.sqrt(np.sum((d - np.sum(d * v, axis=1)[:, np.newaxis] @ v) ** 2))
    return dist
