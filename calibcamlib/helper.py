import numpy as np
import scipy


def get_line_dist(r1, e1, r2, e2):
    n = np.cross(e1, e2)
    return np.abs(np.sum(n * (r1 - r2)))


def null_space_batch(vecs):
    """
    Compute the null space projection matrix for each vector in a batch.
    Returns a batch of (3x3) projection matrices.
    """
    u = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # Normalize vecs
    return np.eye(3) - np.einsum('...i,...j->...ij', u, u)  # Null space projection in 3D space


def intersect(bases, vecs):
    ray_ok = ~np.any([
        np.isnan(bases),
        np.isnan(vecs)
    ], axis=(0, 2))

    bases = bases[ray_ok]
    vecs = vecs[ray_ok]

    if bases.shape[0] < 2:
        return np.full(3, fill_value=np.nan)

    # Compute the null space projection matrices
    M = null_space_batch(vecs)  # Shape: (n, 3, 3)
    # Compute Mbase[u] = M[u] @ bases[u] for all u
    # Sum over all planes
    M_sum = np.sum(M, axis=0)  # Shape: (3, 3)
    # Check rank
    if np.linalg.matrix_rank(M_sum) < 3:
        return np.full(3, fill_value=np.nan)

    Mbase_sum = np.einsum('kij,kj->i', M, bases)  # Shape: (n, 3)
    # Solve for intersection point
    return np.squeeze(np.linalg.solve(M_sum, Mbase_sum))


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
