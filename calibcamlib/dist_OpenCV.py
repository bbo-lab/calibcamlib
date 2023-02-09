# implements full 5 coefficient distortion from OpenCV 4

import numpy as np

def distort(boards_coords_ideal, ks):
    r2 = np.sum(boards_coords_ideal[..., 0:2] ** 2, axis=-1, keepdims=True)
    b = boards_coords_ideal

    def distort_dim(b_d, k_rs, k_ps):
        return (
                b_d * (1 + k_rs[..., (0,)] * r2 + k_rs[..., (1,)] * r2 ** 2 + k_rs[..., (2,)] * r2 ** 3) +
                2 * k_ps[0] * b[..., (0,)] * b[..., (1,)] +
                k_ps[1] * (r2 + 2 * b_d ** 2)
        )

    boards_coords_dist = np.concatenate((
        distort_dim(b[..., (0,)], ks[[0, 1, 4]], ks[[2, 3]]),
        distort_dim(b[..., (1,)], ks[[0, 1, 4]], ks[[3, 2]]),
    ), -1)

    return boards_coords_dist


def distort_inverse(ab_dist, k):
    n = ab_dist.shape[0]
    s = np.sqrt(np.sum(ab_dist ** 2, axis=1))
    r = np.zeros(n)

    for u in np.where(s > 0)[0]:
        rts = np.roots(np.array([k[4], 0, k[1], 0, k[0], 0, 1, -s[u]]))
        rtsind = np.all([np.imag(rts) == 0, rts >= 0], axis=0)
        if not np.any(rtsind):
            r[u] = np.nan
        else:
            r[u] = np.min(np.real(rts[rtsind]))
    ab = ab_dist * (r / s)[:, np.newaxis]

    if np.all(k[2:4] == 0):
        return ab
    else:
        # Warning: There is no closed-form solution for tangential + radial distortion. Thus, we use a solver from
        #  scipy.optimize. If this function is used within a time-critical scope, this is probably slow
        from scipy.optimize import fsolve

        ab_ud = []
        for p_o, p_d in zip(ab, ab_dist):
            sol = fsolve(lambda p: dist_opt_func(p, p_d, k), p_o, full_output=True, maxfev=1000, xtol=1e-6)
            if not sol[2] == 1:
                print(sol[3])
            ab_ud.append(sol[0])

        return np.array(ab_ud)


def dist_opt_func(ab, ab_d, k):
    # This function is included in the unit tests and tested correct
    a = ab[0]
    b = ab[1]
    r2 = (a ** 2 + b ** 2)
    rcoeff = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[4] * r2 ** 3
    return (
        2 * k[2] * a * b + k[3] * (3 * a ** 2 + b ** 2) + a * rcoeff - ab_d[0],
        2 * k[3] * a * b + k[2] * (3 * b ** 2 + a ** 2) + b * rcoeff - ab_d[1]
    )
