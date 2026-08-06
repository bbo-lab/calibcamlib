# implements full 5 coefficient distortion from OpenCV 4

import numpy as np
from bbo import vectorlib

def distort(boards_coords_ideal, ks, fastmath=False):
    xp = vectorlib.get_array_module(boards_coords_ideal)
    b = boards_coords_ideal[..., 0:2] # ignore higher dimensions if any
    bb = xp.square(b)
    r2 = bb[...,(0,)] + bb[...,(1,)]

    # Faster version same accuracy but not bitequal:
    if fastmath:
        b0 = b[..., (0,)]
        b1 = b[..., (1,)]

        if ks[4] == 0:
            polynom = 1 + (ks[0] + ks[1] * r2) * r2
        else:
            polynom = 1 + (ks[0] + (ks[1] + ks[4] * r2) * r2) * r2
        b0b1 = b0 * b1
        #def distort_dim(b_d, k_ps):
        #    return (
        #            b_d * (polynom + k_ps[1] * 2 * b_d) +
        #            k_ps[0] * b0b1 * 2 + #2 * b0 * b1
        #            k_ps[1] * r2    #b0^2 + b1^2
        #    )

        #boards_coords_dist = xp.concatenate((
        #    distort_dim(b0, ks[[2, 3]]),
        #    distort_dim(b1, ks[[3, 2]]),
        #), -1)

        ks0 = ks[[2, 3]][xp.newaxis, :]
        ks1 = ks[[3, 2]][xp.newaxis, :]
        add0 = b * polynom
        add1 = (ks0 * 2) * b0b1
        add2 = ks1 * (r2 + 2 * bb)
        boords_coords_dist = add0 + add1 + add2
        return boords_coords_dist

    def distort_dim(b_d, k_rs, k_ps):
        return (
                b_d * (1 + k_rs[..., (0,)] * r2 + k_rs[..., (1,)] * r2 ** 2 + k_rs[..., (2,)] * r2 ** 3) +
                2 * k_ps[0] * b[..., (0,)] * b[..., (1,)] +
                k_ps[1] * (r2 + 2 * b_d ** 2)
        )

    boards_coords_dist = xp.concatenate((
        distort_dim(b[..., (0,)], ks[[0, 1, 4]], ks[[2, 3]]),
        distort_dim(b[..., (1,)], ks[[0, 1, 4]], ks[[3, 2]]),
    ), -1)

    return boards_coords_dist

def vectorized_roots(polynomials):
    """
    Compute the roots of multiple polynomials given their coefficients, assuming all polynomials
    have the same length and the first coefficients are non-zero.

    Parameters
    ----------
    polynomials : ndarray, shape (M, N)
        A 2D array where each row represents the coefficients of a polynomial.

    Returns
    -------
    roots : ndarray, shape (M, K)
        A 2D array where each row contains the roots of the corresponding polynomial.
        Extra zeros are appended for polynomials with fewer roots than the maximum degree.
    """
    polynomials = np.asarray(polynomials)
    if polynomials.ndim != 2:
        raise ValueError("Input must be a 2D array where each row represents a polynomial.")

    M, N = polynomials.shape

    # Build companion matrices for all polynomials
    companion_matrices = np.zeros((M, N - 1, N - 1), dtype=polynomials.dtype)
    for i in range(N - 2):
        companion_matrices[:, i + 1, i] = 1  # Set sub-diagonal to 1

    companion_matrices[:, 0, :] = -polynomials[:, 1:] / polynomials[:, (0,)]

    # Compute eigenvalues (roots) for all companion matrices
    roots = np.linalg.eigvals(companion_matrices)
    return roots


def distort_inverse(ab_dist:np.ndarray, k):
    n = ab_dist.shape[0]

    s = np.sqrt(np.sum(ab_dist ** 2, axis=1))

    valid_coefficients = []
    valid_indices = np.where(s > 0)[0]
    num_valid = len(valid_indices)
    keep_trailing_zeros = False
    vectorized = True

    # Polynom is -s r^0 + 1 r^1 + 0 r^2 + k1 r^3 ...
    # from s^2 = (x^2 + y^2) (1 + k1 r^2 + k2 r^4 ...)
    valid_coefficients.append(-s[valid_indices])
    valid_coefficients.append(np.ones(num_valid))
    if np.any(k[[0,1,4]] != 0) or keep_trailing_zeros:
        z = np.zeros(num_valid)
        valid_coefficients.append(z)
        valid_coefficients.append(np.full(num_valid, fill_value=k[0]))
        if np.any(k[[1,4]]) != 0 or keep_trailing_zeros:
            valid_coefficients.append(z)
            valid_coefficients.append(np.full(num_valid, fill_value=k[1]))
            if k[4] != 0 or keep_trailing_zeros:
                valid_coefficients.append(z)
                valid_coefficients.append(np.full(num_valid, fill_value=k[4]))

    valid_coefficients = np.stack(valid_coefficients[::-1], axis=1)

    if vectorized:
        roots = vectorized_roots(valid_coefficients)
    else:
        roots = np.array([np.roots(c) for c in valid_coefficients])  # this is very slow

    is_real = np.isreal(roots)
    is_non_negative = roots >= 0
    valid_mask = is_real & is_non_negative

    valid_roots = np.where(valid_mask, np.real(roots), np.inf)

    min_roots = np.min(valid_roots, axis=1)

    r = np.full(n, fill_value=np.nan)
    r[valid_indices] = min_roots

    ab = ab_dist * (r / s)[:, np.newaxis]
    ab[s_0_mask] = ab_dist[s_0_mask]

    if np.all(k[2:4] == 0):
        return ab
    else:
        # Warning: There is no closed-form solution for tangential + radial distortion. Thus, we use a solver from
        #  scipy.optimize. If this function is used within a time-critical scope, this is probably slow
        from scipy.optimize import fsolve
        if distort_inverse.dist_opt_func_compiled is None:
            try:
                from numba import jit
                distort_inverse.dist_opt_func_compiled = jit(nopython=True)(dist_opt_func)
            except ImportError:
                distort_inverse.dist_opt_func_compiled = dist_opt_func

        ab_ud = []
        for p_o, p_d in zip(ab, ab_dist):
            sol = fsolve(distort_inverse.dist_opt_func_compiled, p_o, args=(p_d, k), full_output=True, maxfev=1000, xtol=1e-6)
            # if not sol[2] == 1:
            #     print(sol[3])
            ab_ud.append(sol[0])

        return np.array(ab_ud)


distort_inverse.dist_opt_func_compiled = None

def dist_opt_func(ab, ab_d, k):
    # This function is included in the unit tests and tested correct
    a = ab[0]
    b = ab[1]
    a2 = a ** 2
    b2 = b ** 2
    ab = 2 * a * b
    r2 = a2 + b2
    rcoeff = 1 + r2 * (k[0] + r2 * (k[1] + r2 * k[4]))
    return (
        k[2] * ab + k[3] * (3 * a2 + b2) + a * rcoeff - ab_d[0],
        k[3] * ab + k[2] * (3 * b2 + a2) + b * rcoeff - ab_d[1]
    )