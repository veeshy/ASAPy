from ASAPy import CovManipulation

import scipy.linalg as LA
import numpy as np
import pandas as pd

from warnings import warn

class XsecSampler:
    def __init__(self, h, zaid_1, mt_1, zaid_2=None, mt_2=None):
        """
        Sampling methods for cross-sections
        Parameters
        ----------
        h : pd.HDF5Store
        zaid_1 : int or str
        mt_1 : int or str
        zaid_2 : None or int or str
        mt_2 : None or int or str
        """
        if zaid_2 is None:
            zaid_2 = zaid_1
        if mt_2 is None:
            mt_2 = mt_1

        # load the cov and std_dev from the store
        self.std_dev_df = h['{0}/{1}/{2}/{3}/std_dev'.format(zaid_1, mt_1, zaid_2, mt_2)]
        self.corr_df = h['{0}/{1}/{2}/{3}/corr'.format(zaid_1, mt_1, zaid_2, mt_2)] / 1000

        # correct the correlation for neg eigenvalues if needed
        self.corr_df.loc[:, :] = self._fix_non_pos_semi_def_matrix(self.corr_df.values)

        # calculate the cov
        cov = CovManipulation.correlation_to_cov(self.std_dev_df['s.d.(1)'].values, self.corr_df.values)
        # fix cov in case the correlation calculation introduced neg eigs
        cov = self._fix_non_pos_semi_def_matrix(cov)

        # save the cov to a member var
        cov_df = self.corr_df.copy()
        cov_df.loc[:, :] = cov
        self.cov_df = cov_df

        print('Min eig of cov: {0}'.format(LA.eigvalsh(cov).min()))

    def sample(self, sample_type, n, raise_on_bad_sample=False, remove_neg=True, return_relative=True,
               allow_singular=False):
        """
        Samples using LHS

        Parameters
        ----------
        sample_type : str
            'norm' or 'lhs' to perform multi-variate sampling using these distros
        n : int
            Number of samples
        raise_on_bad_sample : bool
            Option to raise if a negative sample is found
        remove_neg : bool
            Option to remove neg-samples while still trying to sample n
        return_relative : bool
            Option to return relative values (sampled_val / mean)
        allow_singular : bool
            Option to allow singular matrix cov when sampling from norm
        Returns
        -------
        np.array
            Relative sample (sample / original group mean value) (groups x n)
        """

        over_sample_n = n * 2

        if sample_type.lower() == 'lhs':
            samples = CovManipulation.lhs_normal_sample_corr(self.std_dev_df['x-sec(1)'].values,
                                                             self.std_dev_df['s.d.(1)'].values, self.corr_df.values,
                                                             over_sample_n)
        elif sample_type.lower() == 'norm':
            samples = CovManipulation.normal_sample_corr(self.std_dev_df['x-sec(1)'], self.cov_df, over_sample_n,
                                                         allow_singular=allow_singular)

        else:
            raise Exception('Sampling type: {0} not implimented'.format(sample_type))

        if return_relative:
            mean = self.std_dev_df['x-sec(1)'].values
        else:
            mean = np.ones(self.std_dev_df['x-sec(1)'].shape)

        samples = self._sample_check(samples, mean, remove_neg)

        num_samples_worked = samples.shape[1]

        if num_samples_worked < n:
            if not raise_on_bad_sample:
                # try again with bigger n
                samples = self.sample(sample_type, over_sample_n, raise_on_bad_sample=True,
                                      remove_neg=remove_neg, return_relative=return_relative,
                                      allow_singular=allow_singular)
            else:
                raise Exception(
                    'Tried twice to get {0} samples, only was able to make {1}'.format(n / 2, num_samples_worked))

        # grab only n samples
        samples = samples.iloc[:, 0:n]
        # number samples from 0 to n
        samples = samples.T.reset_index(drop=True).T

        return samples

    def _sample_check(self, samples, mean, remove_neg):
        """
        Creates relative sample df keeping the group structure, removing negative samples (if desired)
        """

        sample_df = samples / mean
        sample_df = pd.DataFrame(sample_df.T, index=range(1, len(mean) + 1))

        if remove_neg:
            sample_df = sample_df.loc[:, ((sample_df < 0).sum() == 0)].dropna()

        return sample_df

    def _fix_non_pos_semi_def_matrix(self, corr_matrix):
        """
        Uses eigen-decomposition (m=PDP^-1) to fix non positive semi-definite matricies.
        If all eig > 1e-13, no changes made

        Parameters
        ----------
        corr_matrix : np.array

        References
        ----------
        zhu2015sampling appendix a
        """

        # m = PDP^-1 (np.dot(np.dot(p, np.diag(d)), np.linalg.inv(p)))
        # P is eigenvectors, D is eigenvalues in diag matrix

        # corr_matrix is always symmetric
        # eigs, P = LA.eigh(np.float128(corr_matrix))
        eigs, P = LA.eigh(corr_matrix)

        # replace all negative and zero eigs with a small eps
        bad_index = np.where(eigs <= 1e-8)

        eigs[bad_index] = 1e-8

        # remake the corr matrix with these bad eigenvalues removed
        fixed_corr = np.dot(np.dot(P, np.diag(eigs)), LA.inv(P))

        wtf_eigs = LA.eigvalsh(fixed_corr)
        print('min eig after removing neg', min(wtf_eigs))
        print('min eig desired', min(eigs))

        return fixed_corr


def map_groups_to_continuous(e_sigma, high_e_bins, multi_group_val, max_e=None, min_e=None):
    """
    Maps the grouped multi_group_val onto the sigma based on the continuous energy e_sigma
    Parameters
    ----------
    e_sigma : np.array
        Energy values where the continuous xsec is known in MeV
    high_e_bins : pd.series
        The high energy bins in the group structure in eV
    multi_group_val : pd.series
        multi_group_values with index as the group #
    max_e : float
        Max energy where energies above this, multi_group_val set to zero
    min_e : float
        Min energy where energies below this, multi_group_val set to zero

    Returns
    -------
    np.array
        The multi-group values mapped onto e_sigma
    """
    if max_e is None:
        max_e = max(high_e_bins)
    if min_e is None:
        min_e = min(high_e_bins) / 10
        warn("Min e set to {0} eV since it was not provided".format(min_e))

    num_groups = len(high_e_bins)

    grouped_dev = []
    sorted_e = sorted(high_e_bins.values)
    e_sigma_ev = e_sigma * 1e6

    for e in e_sigma_ev:
        # find where e_cont is in the groups
        if e > max_e or e < min_e:
            # ensure we won't be below or above the bins
            grouped_dev.append(1)
        else:
            # searchsorted will return the index in the e_high list where
            # the e_cont would be if it was placed in the list then sorted
            # this index is the same as the group # we are in
            idx = np.searchsorted(sorted_e, e)
            # convert idx to group # (since we sorted the E from low to high)
            group_num = num_groups - idx + 1  # -> idx 252 is group 1, idx 1 is group 252 (can't have idx 0 by design)
            sd = multi_group_val[group_num]
            grouped_dev.append(sd)

    return np.array(grouped_dev)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pyne import ace

    h = pd.HDFStore('../scale_cov_252.h5', 'r')
    w184_102_std = h['74184/102/74184/102/std_dev']

    libFile = ace.Library('/Users/veeshy/MCNP6/MCNP_DATA/xdata/endf71x/W/74184.710nc')
    libFile.read()
    libFile.find_table('74184')
    a = libFile.tables[list(libFile.tables.keys())[0]]

    e = a.energy
    st = a.find_reaction(102).sigma

    ####

    w = XsecSampler(h, 74184, 102)
    sample_df = w.sample('norm', 500, allow_singular=True, return_relative=False)

    fig, ax = plt.subplots()
    ax.loglog(w184_102_std['e high'], w184_102_std['s.d.(1)'] ** 2, drawstyle='steps-mid', label='Diag(cov) Before')
    ax.loglog(w184_102_std['e high'], np.diag(np.cov(sample_df)), drawstyle='steps-mid', label='Diag(cov) After')

    ax.legend()
    plt.show()

    #### plot cov
    fig, ax = plt.subplots(ncols=3, figsize=(12, 6))
    cax = ax[0].matshow(w.corr_df.values)
    fig.colorbar(cax, ax=ax[0], fraction=0.046, pad=0.04)

    cax = ax[1].matshow(np.corrcoef(sample_df))
    fig.colorbar(cax, ax=ax[1], fraction=0.046, pad=0.04)
    #
    # cax = ax[2].matshow(np.corrcoef(sample_df_norm))
    # fig.colorbar(cax, ax=ax[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()

    ####
    sample_df = w.sample('norm', 500, allow_singular=True, return_relative=True)
    fig, ax = plt.subplots()
    for i in range(30):
        ax.loglog(e * 1e6, map_groups_to_continuous(e, w184_102_std['e high'], sample_df[i],
                                                    min_e=w184_102_std['e low'].min()) * st, label=i)

    ax.loglog(e * 1e6, st, linestyle='-.', color='k')
    ax.set_xlim([180, 190])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    plt.show()