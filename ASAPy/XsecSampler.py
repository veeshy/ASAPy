# from warnings import warn
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as LA
from pyne import ace
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
#
from ASAPy import CovManipulation
from ASAPy import AceIO
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
    h = cov_hdf_store

    # sample data, keep the relative and full values sampled for plotting later
    xsec = XsecSampler(h, zaid, mt)

    sample_df_full_vals = xsec.sample(sample_type, num_samples, return_relative=False, remove_neg=remove_neg)
    mean = xsec.std_dev_df['x-sec(1)'].values
    # get the full values by multiplying in the mean by having the internal sample checker "normalize" the values to the 1/mean

    sample_df = xsec._sample_check(sample_df_full_vals.T, mean, remove_neg=False)

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
    libFile = ace.Library(os.path.expanduser(ace_file))
    libFile.read()
    libFile.find_table(str(zaid))
    xsec_tables = libFile.tables[list(libFile.tables.keys())[0]]

    e = xsec_tables.energy
    st = xsec_tables.find_reaction(mt).sigma

    return e, st

def plot_sampled_info(ace_file, h, zaid, mt, sample_df, sample_df_full_vals, zaid_2=None, mt_2=None, output_base=''):
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
    output_base : str
        Base folder to output at
    """
    # plot the cov
    xsec, corr = XsecSampler.load_zaid_mt(h, zaid, mt, zaid_2, mt_2)

    fig, ax = plt.subplots()
    ax.loglog(xsec['e high'], xsec['s.d.(1)'] ** 2, drawstyle='steps-mid', label='Diag(cov) ENDF')
    ax.loglog(xsec['e high'], np.diag(np.cov(sample_df_full_vals)), drawstyle='steps-mid', label='Diag(cov) Sampled')
    ax.legend()
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Covariance")
    plt.savefig("{2}{3}{0}_{1}_sampled_cov.eps".format(zaid, mt, output_base, os.path.sep), bbox_inches='tight')

    # Plot some xsec
    e, st = get_mt_from_ace(ace_file, zaid, mt)

    fig, ax = plt.subplots()

    # ensure we don't try to plot too many samples
    num_xsec = 20

    # if less than num_xsec samples, plot all of them
    if num_xsec > xsec.shape[1]:
        num_xsec = xsec.shape[1]

    # plot the base x-sec too
    ax.loglog(e * 1e6, st, linestyle='-.', color='k')

    for i in range(num_xsec):
        ax.loglog(e * 1e6, map_groups_to_continuous(e, xsec['e high'], sample_df.iloc[:, i],
                                                    min_e=xsec['e low'].min() - 1) * st, label=i)
    # plot the base again so it appears on top
    ax.loglog(e * 1e6, st, linestyle='-.', color='k')

    ax.legend(['Actual', 'Samples'])

    ax.set_xlabel('Energy (Ev)')
    ax.set_ylabel('Cross Section (b)')

    plt.savefig("{2}{3}{0}_{1}_few_sampled_xsec.eps".format(zaid, mt, output_base, os.path.sep), bbox_inches='tight')

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
    plt.savefig("{2}{3}{0}_{1}_sampled_corr_compare.eps".format(zaid, mt, output_base, os.path.sep), bbox_inches='tight')

    plot_xsec(ace_file, h, zaid, mt, output_base)


def plot_xsec(ace_file, h, zaid, mt, output_base='./', pad_rel_y_decades=False):
    """
    Plots xsec from ACE file and rel deviation from error store for ZAID's mt

    Parameters
    ----------
    ace_file
    h
    zaid
    mt
    output_base : str
        Base path to save plot to

    Returns
    -------

    """
    # plot the xsec + the std_dev
    xsec, corr = XsecSampler.load_zaid_mt(h, zaid, mt, zaid, mt)
    e, st = get_mt_from_ace(ace_file, zaid, mt)
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.3])
    ax = fig.add_subplot(gs[0])
    ax.loglog(e * 1e6, st)
    ax.grid(alpha=0.25)
    # turn off the x labels w/o turning off the grid x
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    ax.set_ylabel('Cross-Section (b)')

    # second plot (rel dev %)
    ax2 = fig.add_subplot(gs[1], sharex=ax)  # the above ax
    ax2.loglog(xsec['e high'], xsec['rel.s.d.(1)'] * 100, drawstyle='steps-mid')

    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('% Rel. Dev.')
    ax2.grid(alpha=0.25)

    if pad_rel_y_decades:
        # ensure this std dev has at least two decades plotted
        # get the base 10 # that log would give back 1.05 -> log10(1.05) = 2.1189e-2, want min to be 10^-2
        # high val, we want the next decade, all with proper scaling
        low_val = xsec['rel.s.d.(1)'].min() * 100
        high_val = xsec['rel.s.d.(1)'].max() * 100 * 10
        low_val = 10**int("{0:e}".format(low_val).split('e')[-1])
        high_val = 10**int("{0:e}".format(high_val).split('e')[-1])

        if low_val == high_val:
            low_val = high_val / 10

        ax2.set_ylim([low_val, high_val])

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    plt.savefig("{2}{3}{0}_{1}_base_xsec_with_std.png".format(zaid, mt, output_base, os.path.sep), bbox_inches='tight', dpi=450)


def write_sampled_data(h, ace_file, zaid, mt, sample_df_rel, output_formatter='xsec_sampled_{0}', zaid_2=None, mt_2=None):
    """
    Write sampled data to files + the group-wise sampled data to a csv

    :param h: the cov h5 handle
    :param ace_file:
    :param sample_df_rel:
    :param output_formatter:
    :return:
    """

    # load the cov to get the mapping
    xsec, corr = XsecSampler.load_zaid_mt(h, zaid, mt, zaid_2, mt_2)

    ae = AceIO.AceEditor(ace_file)
    e = ae.energy
    original_sigma = ae.get_sigma(mt)

    # sample_df contains relative values (sampled / mean) which are then multiplied by the actual continuous xsec and written to an ace file
    for idx, col in sample_df_rel.iteritems():
        # set the sigma in place
        ae.set_sigma(mt, map_groups_to_continuous(e, xsec['e high'], col,
                                                   min_e=xsec['e low'].min()) * original_sigma)
        ae.apply_sum_rules()

        w = AceIO.WriteAce(ace_file)
        w.replace_array(original_sigma, ae.get_sigma(102))
        w.write_ace(output_formatter.format(idx))

    # add in e groups then print all relative data and the base xsec
    sample_df_rel.insert(0, 'e low', xsec['e low'])
    sample_df_rel.insert(1, 'e high', xsec['e high'])
    sample_df_rel.insert(2, 'xsec', xsec['x-sec(1)'])
    sample_df_rel.insert(3, 'std', xsec['rel.s.d.(1)'])
    sample_df_rel.to_csv('{0}.csv'.format(output_formatter.format('samples')))


if __name__ == "__main__":
    import os

    store_name = '../scale_cov_252.h5'
    with pd.HDFStore(store_name, 'r') as h:

        ace_file = '~/MCNP6/MCNP_DATA/xdata/endf71x/U/92235.710nc'
        zaid = 92235
        mt = 18

        # sample_df, sample_df_full = sample_xsec(h, mt, zaid, 1000)
        #
        # plot_sampled_info(ace_file, h, zaid, mt, sample_df, sample_df_full, output_base='../u235_fis/')
        #
        # write_sampled_data(h, ace_file, zaid, mt, sample_df, output_formatter='../u235_fis/u_{0}')
        #
        #

        plot_xsec(ace_file, h, zaid, mt, output_base='./')