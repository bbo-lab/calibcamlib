# implements model0 from MaEtAl2003
import numpy as np
import calibcamlib.dist_OpenCV


def distort(abi, k):
    assert np.all(k[2:] == 0), "Only the first 2 k entries may differ from 0"
    return calibcamlib.dist_OpenCV.distort(abi, k)


def distort_inverse(ab_rd, k):
    assert np.all(k[2:] == 0), "Only the first 2 k entries may differ from 0"
    return calibcamlib.dist_OpenCV.distort_inverse(abi, k)
