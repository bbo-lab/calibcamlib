import numpy as np


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
    ray_ok = ~np.any(np.isnan(bases) | np.isnan(vecs), axis=-1)
    if len(bases.shape) == 2:
        bases = bases[ray_ok]
        vecs = vecs[ray_ok]
        n = bases.shape[0]
        d = bases.shape[1]
        if n < 2:
            return np.full(d, fill_value=np.nan)

        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # Normalize vecs

        M_sum = np.eye(vecs.shape[1]) * n - np.einsum('ki,kj->ij', vecs, vecs)  # Shape: (d, d)
        # Check rank
        if np.linalg.det(M_sum) < 1e-10:
            return np.full(d, fill_value=np.nan)

        Mbase_sum = np.sum(bases,axis=0) - np.dot(np.einsum('ij,ij->i',vecs, bases),vecs)  # Shape: (d)
        return np.linalg.solve(M_sum, Mbase_sum)
    else:
        bases = np.where(ray_ok[..., None], bases, 0)
        vecs = np.where(ray_ok[..., None], vecs, 0)
        vecs_norm = np.linalg.norm(vecs, axis=-1, keepdims=True)
        vecs = np.divide(vecs, vecs_norm, where=vecs_norm > 0)

        valid_counts = np.sum(ray_ok, axis=-1, keepdims=True)
        identity = np.eye(vecs.shape[-1])[None, :, :]  # Shape (1, d, d)
        M_sum = valid_counts[:, None] * identity - np.einsum('mij,mik->mjk', vecs, vecs)

        non_singular_mask = ~(np.linalg.det(M_sum) < 1e-10)

        proj = np.einsum('mij,mij->mi', vecs, bases)[..., None] * vecs
        Mbase_sum = np.sum(bases - proj, axis=1)

        intersections = np.full((M_sum.shape[0], vecs.shape[-1]), np.nan)
        intersections[non_singular_mask] = np.linalg.solve(M_sum[non_singular_mask], Mbase_sum[non_singular_mask])
        return intersections


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
