import warnings

import numpy as np
import SALib.sample.latin as lhs
from scipy.stats import multivariate_normal
from scipy.stats.distributions import norm, uniform
import scipy as sp
from scipy.stats import norm as sci_norm
from numpy.random import uniform as np_uniform
from ASAPy.data.reaction import REACTION_NAME

"""
IC Correlated Sampling from
https://blakeboswell.github.io/article/2016/05/30/mc-parts.html

including: col_independent_matrix, rank, order
"""

def col_independent_matrix(n, d):
    """ return array with shape (n, d) where each
        column is approximately independent
    """
    x = np.arange(1, (n+1))
    p = sp.stats.norm.ppf(x/(n+1))
    p = (p - p.mean()) / p.std()
    score = np.zeros((n, d))
    for j in range(0, score.shape[1]):
        score[:, j] = np.random.permutation(p)
    return score

def rank(arr):
    """ return the rank order of elements in array
    """
    n, k = arr.shape
    rank = np.zeros((n, k))
    for j in range(0, k):
        rank[:, j] = sp.stats.rankdata(arr[:, j], method='ordinal')
    return rank.astype(int) - 1


def order(rank, samples):
    """ order each column of samples according to rank
    """
    n, k = samples.shape
    rank_samples = np.zeros((n, k))
    for j in range(0, k):
        s = np.sort(samples[:, j])
        rank_samples[:, j] = s[rank[:, j]]
    return rank_samples




def correlation_to_cov(std, corr):
    """
    Calculates the cov matrix from correlation matrix + known std_devs
    Parameters
    ----------
    std : np.array
    corr : np.array

    Returns
    -------
    np.array
        The cov matrix

    """

    # given R_{ij} = \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}} and
    # knowing C_ii = (std_dev of variable i)**2 and C_jj = (std_dev of variable j)**2

    cov = np.diag(std**2)

    shape = len(std)
    for i in range(shape):
        for j in range(i+1, shape):
            cov[i, j] = corr[i, j] * std[i] * std[j] #(cov[i, i] * cov[j, j]) ** 0.5

    cov = cov + cov.T - np.diag(np.diag(cov))

    return cov

def cov_to_correlation(cov):

    shape = len(cov)
    corr = np.diag(np.ones(shape))
    for i in range(shape):
        for j in range(i+1, shape):
            corr[i, j] = cov[i, j] / np.abs(cov[i, i] * cov[j, j]) ** 0.5


    corr = corr + corr.T - np.diag(np.diag(corr))
    return corr

_FINFO = np.finfo(float)
_EPS = _FINFO.eps

def gmw_cholesky(A):
    """
    Provides a partial cholesky decomposition that is positive def minus a matrix e

    Return `(P, L, e)` such that P*L = M ->  `MM.T = P.T*A*P = L*L.T + diag(P*e)`

    A snipped for some desired_corr matrix to show that reconstructed matricies agree and how
    gmw cholesky is similar to numpy's cholesky. The intermediate steps are different because
    there is not neccesarily unique lower triangular matricies.

        P = np.linalg.cholesky(desired_corr)
        cholesky_reconstructed = np.dot(P, P.T)

        P, L, e = cm.gmw_cholesky(desired_corr)
        C = np.dot(P, L)
        gmw_cholesky_reconstructed = np.dot(C, C.T)

    Returns
    -------
    P : 2d array
       Permutation matrix used for pivoting.
    L : 2d array
       Lower triangular factor
    e : 1d array
    Positive diagonals of shift matrix `e`.

    Notes
    -----
    The Gill, Murray, and Wright modified Cholesky algorithm.

    Algorithm 6.5 from page 148 of 'Numerical Optimization' by Jorge
    Nocedal and Stephen J. Wright, 1999, 2nd ed.

    This implimentation from https://bitbucket.org/mforbes/pymmf/src/c0028c213c8765e4aa62730e379731b89fcaebff/mmf/math/linalg/cholesky/gmw81.py?at=default

    """
    n = A.shape[0]

    # Test matrix.
    #A = array([[4, 2, 1], [2, 6, 3], [1, 3, -0.004]], Float64)
    #n = len(A)
    #I = identity(n, Float64)

    # Calculate gamma(A) and xi(A).
    gamma = 0.0  # largest value diag
    xi = 0.0  # largest value off-diag
    for i in range(n):
        gamma = max(abs(A[i, i]), gamma)
        for j in range(i+1, n):
            xi = max(abs(A[i, j]), xi)

    # Calculate delta and beta.
    delta = _EPS * max(gamma + xi, 1.0)
    if n == 1:
        beta = np.sqrt(max(gamma, _EPS))
    else:
        beta = np.sqrt(max(gamma, xi / np.sqrt(n**2 - 1.0), _EPS))

    # Initialise data structures.
    a = 1.0 * A
    r = 0.0 * A
    e = np.zeros(n, dtype=float)
    P = np.eye(n, dtype=float)

    # Main loop.
    for j in range(n):
        # Row and column swapping, find the index > j of the largest
        # diagonal element.
        q = j
        for i in range(j+1, n):
            if abs(a[i, i]) >= abs(a[q, q]):
                q = i

        # Interchange row and column j and q (if j != q).
        if q != j:
            # Temporary permutation matrix for swapping 2 rows or columns.
            p = np.eye(n, dtype=float)

            # Modify the permutation matrix P by swapping columns.
            row_P = 1.0*P[:, q]
            P[:, q] = P[:, j]
            P[:, j] = row_P

            # Modify the permutation matrix p by swapping rows (same as
            # columns because p = pT).
            row_p = 1.0*p[q]
            p[q] = p[j]
            p[j] = row_p

            # Permute a and r (p = pT).
            a = np.dot(p, np.dot(a, p))
            r = np.dot(r, p)

        # Calculate dj.
        theta_j = 0.0
        if j < n-1:
            for i in range(j+1, n):
                theta_j = max(theta_j, abs(a[j, i]))
        dj = max(abs(a[j, j]), (theta_j/beta)**2, delta)

        # Calculate e (not really needed!).
        e[j] = dj - a[j, j]

        # Calculate row j of r and update a.
        r[j, j] = np.sqrt(dj)     # Damned sqrt introduces roundoff error.
        for i in range(j+1, n):
            r[j, i] = a[j, i] / r[j, j]
            for k in range(j+1, i+1):
                a[i, k] = a[k, i] = a[k, i] - r[j, i] * r[j, k]     # Keep matrix a symmetric.

    # The Cholesky factor of A.
    return P, r.T, e


def lhs_uniform_sample(num_vars, num_samples):
    """
    Create uncorrelated uniform samples on the unit interval

    Parameters
    ----------
    num_vars
    num_samples

    Returns
    -------

    """
    samples = lhs.sample({'num_vars': num_vars, 'bounds': [[0, 1] for i in range(num_vars)]}, num_samples)

    return samples

def lhs_normal_sample(num_samples, means, std_dev):
    """
    Creates uncorrelated normally distributed sample with mean means and standard deviation std_dev

    Parameters
    ----------
    num_vars
    num_samples
    means
    std_dev

    Returns
    -------

    """
    num_vars = len(means)
    samples = lhs_uniform_sample(num_vars, num_samples)

    for i in range(num_vars):
        # create a norm distro with mean/std_dev then sample from it using percent point func (inv of cdf percentiles)
        samples[:, i] = norm(loc=means[i], scale=std_dev[i]).ppf(samples[:, i])

    return samples

def normal_sample_corr(mean_values, desired_cov, num_samples, allow_singular=False):
    """
    Randomally samples from a normal-multivariate distribution with mean mean_values and cov desired_cov

    Parameters
    ----------
    mean_values
    desired_cov
    num_samples

    Returns
    -------
    np.array

    """

    m = multivariate_normal(mean=mean_values, cov=desired_cov, allow_singular=allow_singular)
    return m.rvs(num_samples)

def sample_with_corr(mean_values, std_dev, desired_corr, num_samples, distro='normal', mt=None):
    """
    Randomally samples from a normal-multivariate distribution using LHS while attempting to get the desired_cov

    Parameters
    ----------
    mean_values
    desired_cov
    num_samples
    distro : str
        norm, lognormal
    Returns
    -------

    """

    mean_values_original = np.array(mean_values)
    desired_corr = np.array(desired_corr)

    # don't sample from mean = 0 cases, or std_dev = 0'ish cases, or when corr on diagonal is zero,
    # which generally happens because the ENDF cov matrix did not have any data for that element and boxer translated
    # it to something that was read in as NAN and then converted to zeros.

    vars_to_not_sample_idx = None

    set_std_dev_below_this_to_zero = 1e-30

    if min(std_dev) <= set_std_dev_below_this_to_zero:
        set_fix_val_idx = std_dev <= set_std_dev_below_this_to_zero
        vars_to_not_sample_idx = list(set_fix_val_idx)

    if min(mean_values) == 0:
        # cannot sample 0 mean with lognormal
        if distro == 'lognormal':
            set_zero_mean_idx = mean_values == 0

            if vars_to_not_sample_idx is not None:
                vars_to_not_sample_idx = [i + j for i,j in zip(vars_to_not_sample_idx, set_zero_mean_idx)]
            else:
                vars_to_not_sample_idx = set_zero_mean_idx

    if np.any(np.diag(desired_corr) == 0):
        set_zero_corr_to_zero = np.diag(desired_corr) == 0
        if vars_to_not_sample_idx is not None:
            vars_to_not_sample_idx = [i + j for i, j in zip(vars_to_not_sample_idx, set_zero_corr_to_zero)]

    # only sample from things that are deemed samplable
    if vars_to_not_sample_idx:
        vars_to_sample = np.invert(vars_to_not_sample_idx)
        mean_values = mean_values[vars_to_sample]
        std_dev = std_dev[vars_to_sample]
        desired_corr = desired_corr[vars_to_sample, :][:, vars_to_sample]

    num_vars = len(mean_values)

    if distro=='lognormal':
        # according to
        # G. Zerovnik, et. al. Transformation of correlation coefficients between normal and lognormal distribution and implications for nuclear applications
        # we must treat the assumed correlations to be normal and transform them to lognormal to sample correclty
        # Large anti and positive correlations along with large relative uncertainties are not allowed for log normal

        print(f"Adjust cov from assumed normal to lognormal via G. Zerovnik for mt={mt} {REACTION_NAME[mt]}")
        log_corr = np.zeros(desired_corr.shape)

        for i in range(len(log_corr)):
            for j in range(len(log_corr)):
                log_corr[i, j] = mean_values[i] * mean_values[j] / (std_dev[i] * std_dev[j]) * (np.exp(desired_corr[i, j] * np.sqrt(np.log(std_dev[i]**2 / mean_values[i]**2 + 1) * np.log(std_dev[j]**2 / mean_values[j]**2 + 1))) - 1)

        with np.warnings.catch_warnings():
            # ignore divide by zero
            np.warnings.simplefilter("ignore")
            diff = np.abs(log_corr - desired_corr) / np.abs(desired_corr)
            diff[np.isnan(diff)] = 0

        diff = diff.sum().sum() / len(log_corr)**2
        print("||(Log(corr) - original) / original||) =", diff)
        desired_corr = log_corr


    # if possible, take cholesky decomposition else do gmw_cholesky with the idea that scipy cholesky is faster than gmw
    try:
        C = sp.linalg.cholesky(desired_corr)
    except np.linalg.LinAlgError:
        P, L, e = gmw_cholesky(desired_corr)
        C = L.T

    M = col_independent_matrix(num_samples, num_vars)
    D = (1. / num_samples) * np.dot(M.T, M)

    try:
        E = sp.linalg.cholesky(D)
    except np.linalg.LinAlgError:
        P, L, e = gmw_cholesky(D)
        E = L.T

    N = np.dot(np.dot(M, np.linalg.inv(E)), C)
    R = rank(N)

    # in theory ANY distro can be used for each var, for easy of interface only one distribution is used here

    distro = distro.lower()

    # create a norm distro with mean/std_dev then sample from it using percent point func (inv of cdf percentiles)
    if distro == 'normal':
        distro_to_sample_from = sci_norm

    elif distro == 'lognormal':
        # using mu/sigma from wiki + the scipy convention of loc and scale to specify the mean and sigma
        mean = [np.log(mean_values[i] / (1 + std_dev[i] ** 2 / mean_values[i] ** 2) ** 0.5) for i in range(num_vars)]
        sigma = [(np.log(1 + std_dev[i] ** 2 / mean_values[i] ** 2)) ** 0.5 for i in range(num_vars)]

        mean_values = np.array(mean)
        std_dev = np.array(sigma)
        distro_to_sample_from = sci_norm

    elif distro == 'uniform':
        distro_to_sample_from = uniform

    else:
        raise Exception("Distro {0} not supported at the moment, though all scipy distros should be usable.".format(distro))


    dists = []
    with warnings.catch_warnings():
        # ignore sampling issues when log due to maybe sampling 0 log values which is supposed to be a constant number
        if distro == 'lognormal':
            warnings.simplefilter("ignore")

        for var_num in range(num_vars):
            # dists.append([distro_to_sample_from.ppf(p, loc=mean_values[var_num], scale=std_dev[var_num]) for p in np_uniform(0.0, 1.0, num_samples)])
            dists.append([distro_to_sample_from.ppf(p, loc=mean_values[var_num], scale=std_dev[var_num]) for p in
                          lhs_uniform_sample(1, num_samples).T[0]])

    dists = np.array(dists)

    # perform any post-processing
    if distro == 'lognormal':
        dists = np.exp(dists)
        dists[np.isnan(dists)] = 1

    dists = order(R, dists.T)

    # ensure 0 mean and 0 std_dev values are not actually sampled
    # if set_zero_mean:
    #     dists[:, set_zero_mean_idx] = 0
    #
    # if set_fixed_val:
    #     dists[:, set_fix_val_idx] = mean_values_original[set_fix_val_idx]

    if vars_to_not_sample_idx:
        dists_with_all_data = np.zeros((num_samples, len(vars_to_not_sample_idx)))
        dists_with_all_data[:, np.invert(vars_to_not_sample_idx)] = dists
        dists_with_all_data[:, vars_to_not_sample_idx] = mean_values_original[vars_to_not_sample_idx]

        dists = dists_with_all_data

    return dists


def __sample_with_corr(mean_values, std_dev, desired_corr, num_samples, distro='normal'):
    """
    Randomally samples from a normal-multivariate distribution using LHS while attempting to get the desired_cov

    Parameters
    ----------
    mean_values
    desired_cov
    num_samples
    distro : str
        normal, lognormal (no proper handling of corr conversion)
    Returns
    -------

    """

    # raise Exception("This method is deprecated please use sample_with_corr")

    # draw samples in an uncorrelated manner
    num_vars = len(mean_values)
    samples = lhs_normal_sample(num_samples, np.zeros(num_vars), np.ones(num_vars))
    # samples = lhs_uniform_sample(num_vars, num_samples)

    # cholesky-like decomp for non PD matricies.
    T = np.corrcoef(samples.T)
    # this decomposition might be right but it's used wrong..
    # permutation, Q, e = gmw_cholesky(T)

    Q = np.linalg.cholesky(T)

    # this matrix has the same correlation as the desired RStar.
    # It is known to be PD since any neg eigenvalues were removed already.
    # this can be changed to using gmw_cholesky to be more general though.
    P = np.linalg.cholesky(desired_corr)

    dependent_samples = np.dot(samples, np.dot(P, np.linalg.inv(Q)).T)

    # for il=1:ntry
    #     for j=1:nvar
    #         % rank RB
    #         [r,id]=ranking(RB(:,j));
    #         % sort R
    #         [RS,id]=sort(R(:,j));
    #         % permute RS so has the same rank as RB
    #         z(:,j) = RS(r).*xsd(j)+xmean(j);
    #     end
    #     ae=sum(sum(abs(corrcoef(z)-corr)));
    #     if(ae<amin)
    #         zb=z;
    #         amin=ae;
    #     end
    # end

    ntry = 1
    amin = 1.8e308
    z = np.zeros(np.shape(samples))
    for il in range(ntry):
        for j in range(num_vars):
            r = np.argsort(dependent_samples[:, j])
            rank = np.zeros(np.shape(r), dtype=int)
            rank[r] = np.array(range(num_samples))
            rs = np.sort(samples[:, j])
            z[:, j] = np.multiply(rs[rank], std_dev[j]) + mean_values[j]

        ae = np.abs(np.corrcoef(z.T) - desired_corr).sum().sum()

        if ae < amin:
            zb = z
            amin = ae
        else:
            raise Exception('Could not order samples ae={0}'.format(ae))

    # zb are the uniform correlated samples, now transform them to desired
    #
    # transform the uniform sample about the mean to the unit interval
    for i in range(num_vars):
        zb[:, i] = (zb[:, i] - min(zb[:, i]))
        zb[:, i] = zb[:, i] / max(zb[:, i])

        slightly_lt0 = zb[:, i] <= 0.0  # + 1e-5
        slightly_gt1 = zb[:, i] >= 1.0  # - 1e-5

        zb[slightly_lt0, i] = 1e-10  # 1e-5
        zb[slightly_gt1, i] = 1-1e-10  # 1.0 - 1e-5

    distro = distro.lower()

    # using the desired distro's ppf, sample the distro with the correlated uniform sample
    for i in range(num_vars):
        # create a norm distro with mean/std_dev then sample from it using percent point func (inv of cdf percentiles)
        if distro == 'normal':
            zb[:, i] = norm.ppf(zb[:, i], loc=mean_values[i], scale=std_dev[i])
        elif distro == 'lognormal':
            # using mu/sigma from wiki + the scipy convention of loc and scale to specify the mean and sigma
            mean = np.log(mean_values[i] / (1 + std_dev[i]**2/mean_values[i]**2)**0.5)
            sigma = (np.log(1 + std_dev[i]**2/mean_values[i]**2))**0.5
            zb[:, i] = np.exp(norm.ppf(zb[:, i], loc=mean, scale=sigma))
        elif distro == 'uniform':
            zb[:, i] = uniform.ppf(zb[:, i], loc=mean_values[i], scale=std_dev[i])
        else:
            raise Exception("Distro {0} not supported at the moment".format(distro))

    return zb

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    #
    # desired_corr = np.diag([1]*25) + np.diag([-0.5]*24, 1) + np.diag([-0.5]*24, -1)
    #
    # dependent_samples = lhs_normal_sample_corr(np.array(np.ones(25)*20), np.ones(25)*0.05*20, desired_corr, 500)
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(dependent_samples[4, :])
    # plt.show()
    #
    # # correlation is good..
    # plt.imshow(np.corrcoef(dependent_samples.T))
    # plt.colorbar()
    # plt.show()
    # a = np.array([[4, 2, 1], [2, 6, 3], [1, 3, -0.004]])
    # p,r,e = gmw_cholesky(a)
    # #print(np.dot(np.dot(p, a), p.T))
    # m = np.dot(p, r)
    # print(np.dot(m, m.T))
    pass