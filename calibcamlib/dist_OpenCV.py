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


def distort_inverse(ab_rd, k):
    assert np.all(k[2:4] == 0), 'This needs to be implemented'
    n = ab_rd.shape[0]
    s = np.sqrt(np.sum(ab_rd ** 2, axis=1))
    r = np.zeros(n)

    for u in np.where(s > 0)[0]:
        rts = np.roots(np.array([k[4], 0, k[1], 0, k[0], 0, 1, -s[u]]))
        rtsind = np.all([np.imag(rts) == 0, rts >= 0], axis=0)
        if not np.any(rtsind):
            r[u] = np.nan
        else:
            r[u] = np.min(np.real(rts[rtsind]))

    return ab_rd * (r / s)[:, np.newaxis]
