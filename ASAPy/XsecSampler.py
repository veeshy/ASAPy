# from warnings import warn
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as LA
from pyne import ace
#
from ASAPy import CovManipulation
#

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

        # load the cov and std_dev from the store
        self.std_dev_df, self.corr_df = self.load_zaid_mt(h, zaid_1, mt_1, zaid_2, mt_2)

        # correct the correlation for neg eigenvalues if needed
        self.corr_df.loc[:, :] = self._fix_non_pos_semi_def_matrix_eigen(self.corr_df.values)

    @staticmethod
    def load_zaid_mt(h, zaid_1, mt_1, zaid_2=None, mt_2=None):
        """
        Loads the relevant std dev and corr df's
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
        std_dev_df = h['{0}/{1}/{2}/{3}/std_dev'.format(zaid_1, mt_1, zaid_2, mt_2)]
        corr_df = h['{0}/{1}/{2}/{3}/corr'.format(zaid_1, mt_1, zaid_2, mt_2)] / 1000

        return std_dev_df, corr_df

    def calc_cov(self):
        """
        Calculates the cov from corr and std_dev. May cause matrix to become non-pos def

        Returns
        -------
        pd.DataFrame
            Same info as corr df
        """
        cov = CovManipulation.correlation_to_cov(self.std_dev_df['s.d.(1)'].values, self.corr_df.values)
        cov_df = self.corr_df.copy()
        cov_df.loc[:, :] = cov

        return cov_df



    def sample(self, sample_type, n, raise_on_bad_sample=False, remove_neg=True, return_relative=True,
               set_neg_to_zero=False):
        """
        Samples using LHS

        Parameters
        ----------
        sample_type : str
            'norm' or 'lognorm' to perform multi-variate sampling using these distros
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

        if sample_type.lower() == 'norm':
            samples = CovManipulation.lhs_normal_sample_corr(self.std_dev_df['x-sec(1)'].values,
                                                             self.std_dev_df['s.d.(1)'].values, self.corr_df.values,
                                                             over_sample_n, distro='norm')
        elif sample_type.lower() == 'lognorm':
            samples = CovManipulation.lhs_normal_sample_corr(self.std_dev_df['x-sec(1)'].values,
                                                             self.std_dev_df['s.d.(1)'].values, self.corr_df.values,
                                                             over_sample_n, distro='lognorm')
        else:
            raise Exception('Sampling type: {0} not implimented'.format(sample_type))

        if return_relative:
            mean = self.std_dev_df['x-sec(1)'].values
        else:
            mean = np.ones(self.std_dev_df['x-sec(1)'].shape)

        samples = self._sample_check(samples, mean, remove_neg)


        if set_neg_to_zero:
            samples[samples < 0] = 0
        else:
            num_samples_worked = samples.shape[1]
            # over sample to make sure we don't get negative samples.
            # todo: this should be reworked to not oversample in log normal, or when negative samples are fine
            if num_samples_worked < n:
                if not raise_on_bad_sample:
                    # try again with bigger n
                    samples = self.sample(sample_type, over_sample_n, raise_on_bad_sample=True,
                                          remove_neg=remove_neg, return_relative=return_relative)
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

    def _fix_non_pos_semi_def_matrix_eigen(self, corr_matrix):
        """
        Uses eigen-decomposition (m=PDP^-1) to fix non positive semi-definite matricies.
        If all eig > 1e-8, no changes made

        Parameters
        ----------
        corr_matrix : np.array

        References
        ----------
        zhu2015sampling appendix a
        """

        eigs, P = LA.eigh(corr_matrix)
        print('eig_replace: got eig', min(eigs))
        # replace all negative and zero eigs with a small eps
        bad_index = np.where(eigs <= 1e-8)

        # set to some small number
        eigs[bad_index] = min(1e-8 * max(eigs), 1e-8)

        # remake the corr matrix with these bad eigenvalues removed
        fixed_corr = np.dot(np.dot(P, np.diag(eigs)), LA.inv(P))
        print('eig_replace: created eig', min(LA.eigvals(fixed_corr)))

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

            # do not sample if above the cov bins
            if group_num > num_groups:
                grouped_dev.append(1)
                continue

            if group_num < 1:
                grouped_dev.append(1)
                continue

            sd = multi_group_val[group_num]
            grouped_dev.append(sd)

    return np.array(grouped_dev)


def sample_xsec(cov_hdf_store, mt, zaid, num_samples, sample_type='lognorm', remove_neg=False):
    """
    Samples the cov store for cross-section values based on the mat_num, mt, and sample type

    Parameters
    ----------
    cov_hdf_store : pd.HDFStore
        The SCALE/TENDL store containing cov info with keys like 92235/102/92235/102/std_dev
    mt : int
        The reaction MT number to sample
    zaid : int
        The ZAID for the cross-section to sample
    num_samples : int
    sample_type : str
        'lognorm' or 'norm' for sampling the data
    remove_neg : boolean
        Flag to remove samples if they are negative (Removes full samples not just sets neg to zero)

    Returns
    pd.DataFrame
        The sampled df, relative values
    pd.DataFrame
        The sampled df, full values
    -------

    """
    #cov_hdf_store = 'scale_cov_252.h5'
    h = pd.HDFStore(cov_hdf_store, 'r')

    # sample data, keep the relative and full values sampled for plotting later
    xsec = XsecSampler(h, zaid, mt)

    sample_df = xsec.sample(sample_type, num_samples, return_relative=True, remove_neg=remove_neg)
    mean = xsec.std_dev_df['x-sec(1)']
    # get the full values by multiplying in the mean by having the internal sample checker "normalize" the values to the 1/mean

    sample_df_full_vals = xsec._sample_check(sample_df, 1/mean, remove_neg=False)

    return sample_df, sample_df_full_vals

def get_mt_from_ace(ace_file, zaid, mt):
    """
    Loads mt from base_ace file
    Parameters
    ----------
    ace_file : str
        Path to ace file
    zaid : int
        The ZAID in the ace file
    mt : int
        The MT reaction number

    Returns
    -------
    np.array
        Energy values
    np.array
        Cross-section values
    """

    # base_ace = './xe135m/Xe135m-n.ace.txt'
    libFile = ace.Library(ace_file)
    libFile.read()
    libFile.find_table(str(zaid))
    xsec_tables = libFile.tables[list(libFile.tables.keys())[0]]

    e = xsec_tables.energy
    st = xsec_tables.find_reaction(mt).sigma

    return e, st

def plot_sampled_info(ace_file, h, zaid, mt, sample_df_full_vals, zaid_2=None, mt_2=None):
    """
    Plots diag of cov, a few xsecs, and full corr matrix of the sampled data
    Parameters
    ----------
    ace_file
    h
    zaid
    mt
    sample_df_full_vals
    zaid_2
    mt_2

    Returns
    -------

    """
    # plot the cov
    xsec, corr = XsecSampler.load_zaid_mt(h, zaid, mt, zaid_2, mt_2)

    fig, ax = plt.subplots()
    ax.loglog(xsec['e high'], xsec['s.d.(1)'] ** 2, drawstyle='steps-mid', label='Diag(cov) ENDF')
    ax.loglog(xsec['e high'], np.diag(np.cov(sample_df_full_vals)), drawstyle='steps-mid', label='Diag(cov) Sampled')
    ax.legend()
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross-Section (b)")
    plt.savefig("{0}_{1}_sampled_cov.eps".format(zaid, mt), bbox_inches='tight')

    # Plot some xsec
    e, st = get_mt_from_ace(ace_file, zaid, mt)

    fig, ax = plt.subplots()
    num_xsec = 20
    # ensure we don't try to plot too many
    if num_xsec > len(xsec):
        num_xsec = len(xsec)

    for i in range(num_xsec):
        ax.loglog(e * 1e6, map_groups_to_continuous(e, xsec['e high'], sample_df_full_vals[i],
                                                    min_e=xsec['e low'].min() - 1) * st, label=i)
    # plot the base x-sec too
    ax.loglog(e * 1e6, st, linestyle='-.', color='k')

    ax.set_xlabel('Energy (Ev)')
    ax.set_ylabel('Cross Section (b)')

    plt.savefig("{0}_{1}_few_sampled_xsec.eps".format(zaid, mt), bbox_inches='tight')

    # Plot the corr matrix
    fig, ax = plt.subplots(ncols=2, figsize=(8, 6))
    cax = ax[0].matshow(corr.values)
    fig.colorbar(cax, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_xlabel('Group')
    ax[0].xaxis.set_label_position('top')
    ax[0].set_ylabel('Group')

    ax[1].matshow(np.corrcoef(sample_df_full_vals))
    fig.colorbar(cax, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_xlabel('Group')
    ax[1].xaxis.set_label_position('top')
    ax[1].set_ylabel('Group')

    fig.tight_layout()
    plt.savefig("{0}_{1}_sampled_corr_compare.eps".format(zaid, mt), bbox_inches='tight')


if __name__ == "__main__":

    store_name = '../scale_cov_252.h5'
    with pd.HDFStore(store_name, 'r') as h:

        ace_file = '../xe135m/Xe135m-n.ace.txt'
        zaid = 5459
        mt = 102

        sample_df, sample_df_full = sample_xsec(store_name, mt, zaid, 500)
        plot_sampled_info(ace_file, h, zaid, mt, sample_df_full)