import numpy as np
from scipy.linalg import sqrtm

def wasserstein_dist_gaussian(m1, v1, m2, v2):
    """
    symmetric KLD for Multivariate Gaussian distribution
    m: mean (dim,)
    v: covariance (dim, dim)
    """
    d_2 = np.sum((m1 - m2)**2)
    v1_sqrt = sqrtm(v1)
    d_2 += np.trace(v1 + v2 - 2 * sqrtm(v1_sqrt.dot(v2).dot(v1_sqrt)))
    return np.real(d_2)

def sqrtm_A(A):
    u, s, vh = np.linalg.svd(A)
    sqrt_numpy = u.dot(np.eye(s.shape[0]) * s).dot(u.T)
    return sqrt_numpy

def wasserstein_dist_gaussian_A(m1, A1, m2, A2):
    """
    symmetric KLD for Multivariate Gaussian distribution
    m: mean (dim,)
    A: Cholesky decom result of covariance (dim, dim)
    """
    u1, s1, vh1 = np.linalg.svd(A1)
    s1 = s1 * np.eye(s1.shape[0])
    u2, s2, vh2 = np.linalg.svd(A2)
    s2 = s2 * np.eye(s2.shape[0])
    A_prime = u1.dot(s1).dot(u1.T).dot(u2.dot(s2))
    sqrt_term = sqrtm_A(A_prime)
    v1 = A1.dot(A1.T)
    v2 = A2.dot(A2.T)
    
    d_2 = np.sum((m1 - m2)**2)
    d_2 += np.trace(v1 + v2 - 2 * sqrt_term)
    assert d_2.dtype == np.float64
    return d_2
