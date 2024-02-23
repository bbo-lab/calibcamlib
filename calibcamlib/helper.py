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


def calc_3derr(X, P, V) -> (float, np.ndarray):
    """Return the sum of the squared distances of points from lines
    :param X: N_POINTSx3 array of points
    :param P: N_LINESx3 array of points on the lines
    :param V: N_LINESx3 array of line directions
    :return: float of the sum of the squared distances
    :return: (N_POINTS, N_LINES,) array of distances
    """
    dists = np.zeros(shape=(X.shape[0], P.shape[0]))

    for i, p in enumerate(P):
        dists[:, i] = calc_min_line_point_dist(X, p, V[i])

    return np.nansum(dists ** 2), dists


def calc_min_line_point_dist(x, p, v) -> np.ndarray:
    """Return the minimum distances of points from a line
    :param x: Nx3 array of points
    :param p: (3,) array of a point on the line
    :param v: (3,) array of the direction of the line
    :return: (N,) array of distances
    """

    p = p[np.newaxis, :]
    v = v[np.newaxis, :]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    d = x - p
    proj_vecs = d - np.sum(d * v, axis=1, keepdims=True) * v
    dists = np.linalg.norm(proj_vecs, axis=1)
    return dists
