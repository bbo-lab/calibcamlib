import numpy as np
import scipy


def get_line_dist(r1, e1, r2, e2):
    n = np.cross(e1, e2)
    return np.abs(np.sum(n * (r1 - r2)))




def intersect(bases, vecs):
    """
    Compute the intersection point of a set of lines in an arbitrary-dimensional space.

    Each line is defined by a base point and a direction vector.
    The function calculates the intersection point by projecting the base points onto
    the null spaces of the direction vectors, then solving the resulting linear system.

    Parameters:
    -----------
    bases : numpy.ndarray
        An array of shape (n, d) where each row represents the base point of a line in d-dimensional space.
    vecs : numpy.ndarray
        An array of shape (n, d) where each row represents the direction vector of a line in d-dimensional space.

    Returns:
    --------
    numpy.ndarray
        A 1D array of shape (d,) representing the intersection point of the lines.
        If the input data is invalid, or if the lines do not intersect uniquely,
        the function returns an array filled with NaN.
    """
    bases = np.asarray(bases)
    vecs = np.asarray(vecs)
    ray_ok = ~np.any([
        np.isnan(bases),
        np.isnan(vecs)
    ], axis=(0, 2))

    bases = bases[ray_ok]
    vecs = vecs[ray_ok]
    n = bases.shape[0]
    d = bases.shape[1]
    if n < 2:
        return np.full(d, fill_value=np.nan)

    # Compute the null space projection matrices
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # Normalize vecs
    M = np.einsum('...i,...j->...ij', vecs, vecs)  # Null space projection in 3D space

    # Compute Mbase[u] = M[u] @ bases[u] for all u
    # Sum over all planes
    M_sum = np.eye(vecs.shape[1]) * n - np.sum(M, axis=0)  # Shape: (d, d)
    # Check rank
    if np.linalg.matrix_rank(M_sum) < d:
        return np.full(d, fill_value=np.nan)

    Mbase_sum = np.sum(bases,axis=0) - np.einsum('kij,kj->i', M, bases)  # Shape: (d)
    # Solve for intersection point
    return np.linalg.solve(M_sum, Mbase_sum)


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
