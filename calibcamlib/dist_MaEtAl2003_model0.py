# implements model0 from MaEtAl2003

import numpy as np


def distort(abi, k):
    s2 = np.sum(abi ** 2, axis=1)
    return (1 + k[0] * s2 + k[1] * s2 ** 2)[:, np.newaxis] * abi


def distort_inverse(ab_rd, k):
    n = ab_rd.shape[0]
    s = np.sqrt(np.sum(ab_rd ** 2, axis=1))
    r = np.zeros(n)

    for u in np.where(s > 0)[0]:
        rts = np.roots(np.array([k[1], 0, k[0], 0, 1, -s[u]]))
        rtsind = np.all([np.imag(rts) == 0, rts >= 0], axis=0)
        if not np.any(rtsind):
            r[u] = np.nan
        else:
            r[u] = np.min(np.real(rts[rtsind]))

    return ab_rd * (r / s)[:, np.newaxis]
