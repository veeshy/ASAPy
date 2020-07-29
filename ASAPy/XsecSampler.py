import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as LA
from matplotlib import gridspec, cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
import warnings

from ASAPy import CovManipulation
from ASAPy import AceIO

import mpi4py.MPI

#find out which number processor this particular instance is,
#and how many there are in total
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# a created diverging color map from red to green with white in middle
top = cm.get_cmap('Reds', 128)
bottom = cm.get_cmap('Greens', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128))[-1::-1],
                     bottom(np.linspace(0, 1, 128))))
# ensure middle value is actually white
newcolors[128] = (1, 1, 1, 1)
cm_RdGr = ListedColormap(newcolors, name='RdGn')

class XsecSampler:
    def __init__(self, h, zaid_1, mt_1, zaid_2=None, mt_2=None, remove_negative_eig=True):
        """
        Sampling methods for cross-sections
        Parameters
        ----------
        h : pd.HDF5Store
        zaid_1 : int or str
        mt_1 : int or str
        zaid_2 : None or int or str
        mt_2 : None or int or str
        remove_negative_eig : bool
            Replaces any negative eigenvalues in correlation matrix with a small positive number, used to help decomposition during sampling
        """

        # load the cov and std_dev from the store
        self.std_dev_df, self.corr_df = self.load_zaid_mt(h, zaid_1, mt_1, zaid_2, mt_2)

        # correct the correlation for neg eigenvalues if needed
        if remove_negative_eig:
            self.corr_df.loc[:, :] = self._fix_non_pos_semi_def_matrix_eigen(self.corr_df.values)

    @staticmethod
    def load_zaid_mt(h, zaid_1, mt_1, zaid_2=None, mt_2=None, fill_corr_diag=False):
        """
        Loads the relevant std dev and corr df's
        Parameters
        ----------
        h : pd.HDF5Store
        zaid_1 : int or str
        mt_1 : int or str
        zaid_2 : None or int or str
        mt_2 : None or int or str
        fill_corr_diag : bool
            Sets diag of correlation matrix to 1.0 in case original format did not do this
        """
        if zaid_2 is None:
            zaid_2 = zaid_1
        if mt_2 is None:
            mt_2 = mt_1

        # load the cov and std_dev from the store
        std_dev_df = h['{0}/{1}/{2}/{3}/std_dev'.format(zaid_1, mt_1, zaid_2, mt_2)]
        corr_df = h['{0}/{1}/{2}/{3}/corr'.format(zaid_1, mt_1, zaid_2, mt_2)]
        # ensure diagonals and matrix normalized to 1.0 + have 1.0 on the diag
        if fill_corr_diag:
            np.fill_diagonal(corr_df.values, max(np.diag(corr_df.values)))  # boxer format used 1000 as 1.0 for correlations
        corr_df = corr_df / max(np.diag(corr_df.values))


        # make sure e low and e high are in the correct order
        # todo: fix: this is a dumb workaround for my parsed scale cov which flipped e high and e low keys
        if std_dev_df['e low'][1] > std_dev_df['e high'][1]:
            # swap e low and e high values
            temp_e = list(std_dev_df['e low'].values)
            std_dev_df['e low'] = list(std_dev_df['e high'])
            std_dev_df['e high'] = temp_e

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



    def sample(self, sample_type, num_samples, raise_on_bad_sample=False, remove_neg=True, return_relative=True,
               set_neg_to_zero=False):
        """
        Samples using LHS

        Parameters
        ----------
        sample_type : str
            'norm' or 'lognorm' or 'uncorrelated' to perform multi-variate sampling using these distros
        num_samples : int
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

        if sample_type.lower() == 'norm':
            samples = CovManipulation.sample_with_corr(self.std_dev_df['x-sec(1)'].values,
                                                       self.std_dev_df['s.d.(1)'].values, self.corr_df.values,
                                                       num_samples, distro='norm')
        elif sample_type.lower() == 'lognorm':
            samples = CovManipulation.sample_with_corr(self.std_dev_df['x-sec(1)'].values,
                                                       self.std_dev_df['s.d.(1)'].values, self.corr_df.values,
                                                       num_samples, distro='lognorm')
        elif sample_type.lower() == 'uncorrelated':
            # use a correlation of diag(1)
            samples = CovManipulation.sample_with_corr(self.std_dev_df['x-sec(1)'].values,
                                                       self.std_dev_df['s.d.(1)'].values, np.diag(np.ones(len(self.std_dev_df['x-sec(1)'].values))),
                                                       num_samples, distro='norm')
        elif sample_type.lower() == 'uniform':
            samples = CovManipulation.sample_with_corr(self.std_dev_df['x-sec(1)'].values,
                                                       self.std_dev_df['s.d.(1)'].values, self.corr_df.values,
                                                       num_samples, distro='uniform')
        elif sample_type.lower() == 'loguncorrelated':
            samples = CovManipulation.sample_with_corr(self.std_dev_df['x-sec(1)'].values,
                                                       self.std_dev_df['s.d.(1)'].values,
                                                       np.diag(np.ones(len(self.std_dev_df['x-sec(1)'].values))),
                                                       num_samples, distro='lognorm')

        else:
            raise Exception('Sampling type: {0} not implimented'.format(sample_type))

        if return_relative:
            mean = self.std_dev_df['x-sec(1)'].values
        else:
            mean = np.ones(self.std_dev_df['x-sec(1)'].shape)

        samples = self._sample_check(samples, mean, remove_neg)


        if set_neg_to_zero:
            samples[samples < 0] = 0

        num_samples_worked = samples.shape[1]

        if num_samples_worked < num_samples:
            # if not raise_on_bad_sample:
            #     samples = self.sample(sample_type, num_samples, raise_on_bad_sample=True,
            #                           remove_neg=remove_neg, return_relative=return_relative)
            # else:
            if raise_on_bad_sample:
                raise Exception("Could not generate samples due to negative values created. Made {0} of the desired {1}".format(num_samples_worked, num_samples))

        # number samples from 0 to num_samples
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
        if min(eigs) <= 1e-8:
            # replace all negative and zero eigs with a small eps
            bad_index = np.where(eigs <= 1e-8)
            print('eig_replace: got min eig', min(eigs), 'with', len(bad_index), ' eigenvalue less than 1e-8')

            # set to some small number
            eigs[bad_index] = min(1e-8 * max(eigs), 1e-8)

            # remake the corr matrix with these bad eigenvalues removed
            fixed_corr = np.dot(np.dot(P, np.diag(eigs)), LA.inv(P))
            print('eig_replace: created eig', min(LA.eigvals(fixed_corr)))
        else:
            fixed_corr = corr_matrix
            print('No eigen replacement needed for corr matrix')

        return fixed_corr


def map_groups_to_continuous(e_sigma, high_e_bins, multi_group_val, max_e=None, min_e=None, value_outside_of_range=1):
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
        Max energy (eV) where energies above this, multi_group_val set to 1.0 (no sampling).
        Energies below this max_e and above the max(high_e_bins) are treated as if they are in the highest energy bin)
    min_e : float
        Min energy (eV) where energies below this, multi_group_val set to 1.0 (no sampling)
        Energies above this min_e and below the min(high_e_bins) are treated as if they are in the lowest energy bin)
    value_outside_of_range : float
        Value to fill if the mapping would be out of the range [min_e, max_e]

    Returns
    -------
    np.array
        The multi-group values mapped onto e_sigma
    """
    if max_e is None:
        max_e = max(high_e_bins)
    if min_e is None:
        min_e = min(high_e_bins) / 10
        warnings.warn("Min e set to {0} eV since it was not provided".format(min_e))

    if np.equal(min_e, min(high_e_bins.values)):
        raise Exception("Min e bin should be lower than the lowest high_e_bin")

    num_groups = len(high_e_bins)

    # create bins so that they are actually bins and not just end points of the bins by adding the lowest end point
    e_bins_to_map_to = list(high_e_bins.values)
    e_bins_to_map_to.append(min_e)
    if not np.equal(e_bins_to_map_to[0], max_e):
        if max_e < e_bins_to_map_to[0]:
            raise Exception("Max e bin should be greater than or equal to the max high_e_bin")

        e_bins_to_map_to[0] = max_e

    sorted_e = sorted(e_bins_to_map_to)
    e_sigma_ev = np.array(e_sigma)

    # locate where the e to map from is below the min and max bounds so they can be changed to the desired value after searching for indicies
    bins_too_low = np.where(e_sigma_ev < min_e)
    bins_too_high = np.where(e_sigma_ev > max_e)

    bins = np.searchsorted(sorted_e, e_sigma_ev)

    # map to actual group number
    group_nums = 1 + num_groups - bins
    # for now replace the too low and too high bins with nearest good bin. Later they will be overwritten to user value
    # this makes it easier to index indo multi_group_val
    group_nums[bins_too_low] = num_groups
    group_nums[bins_too_high] = 1

    std_dev_mapped_to_e_groups = multi_group_val[group_nums].values

    std_dev_mapped_to_e_groups[bins_too_low] = value_outside_of_range
    std_dev_mapped_to_e_groups[bins_too_high] = value_outside_of_range

    # old logic for grouping
    # for e in e_sigma_ev:
    #     # find where e_cont is in the groups
    #
    #     # if above (below) max (min), assume no std_dev will be mapped
    #     if e > max_e or e < min_e:
    #         grouped_dev.append(value_outside_of_range)
    #
    #     else:
    #         # searchsorted will return the index in the e_high list where
    #         # the e_cont would be if it was placed in the list then sorted
    #         # this index is the same as the group # we are in
    #         idx = np.searchsorted(sorted_e, e)
    #         # convert idx to group # (since we sorted the E from low to high)
    #         group_num = num_groups - idx + 1  # -> idx 252 is group 1, idx 1 is group 252 (can't have idx 0 by design)
    #
    #         # if we are in a bin that is above the min e (below the max e), sample given the min e bin (max e bin)
    #         if group_num > num_groups:
    #             group_num = num_groups
    #
    #         if group_num < 1:
    #             group_num = 1
    #
    #         sd = multi_group_val[group_num]
    #         grouped_dev.append(sd)


    return std_dev_mapped_to_e_groups

def sample_xsec(cov_hdf_store, mt, zaid, num_samples, sample_type='lognorm', remove_neg=False,
                raise_on_bad_sample=False, num_samples_to_make=None):
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
        'lognorm', 'norm', 'uncorrelated' for sampling the data
    remove_neg : boolean
        Flag to remove samples if they are negative (Removes full samples not just sets neg to zero)

    Returns
    pd.DataFrame
        The sampled df, relative values
    pd.DataFrame
        The sampled df, full values
    -------

    """

    if not num_samples_to_make:
        # todo actually use this..
        num_samples_to_make = num_samples

    # ensure type
    num_samples = int(num_samples)

    h = cov_hdf_store

    # sample data, keep the relative and full values sampled for plotting later
    xsec = XsecSampler(h, zaid, mt)

    sample_df_full_vals = xsec.sample(sample_type, num_samples, return_relative=False, remove_neg=remove_neg,
                                      raise_on_bad_sample=raise_on_bad_sample)
    mean = xsec.std_dev_df['x-sec(1)'].values
    # get the full values by multiplying in the mean by having the internal sample checker "normalize" the values to the 1/mean

    sample_df = xsec._sample_check(sample_df_full_vals.T, mean, remove_neg=False)

    # if a NaN is found at this point, it's because the sample_df (rel) calc divided by 0 mean,
    # set the rel dev to the value sampled + 1.0. Later we can check if mean = 0 and if it is just use rel dev
    sample_df[sample_df.isna()] = sample_df_full_vals[sample_df.isna()]

    return sample_df, sample_df_full_vals

def get_mt_from_ace(ace_file, mt):
    """
    Loads mt from base_ace file
    Parameters
    ----------
    ace_file : str
        Path to ace file
    mt : int
        The MT reaction number

    Returns
    -------
    np.array
        Energy values
    np.array
        Cross-section values
    """

    ace_path = os.path.expanduser(ace_file)
    non_typical_xsec = [452, 1018]
    if mt in non_typical_xsec:

        libFile = AceIO.AceEditor(ace_path)
        if mt == 452:
            # got an atypical mt
            e, st = libFile.get_nu_distro()
        elif mt == 1018:
            _, e, st, _ = libFile.get_chi_distro()

    else:
        libFile = AceIO.AceEditor(ace_path)
        e = libFile.get_energy(mt)
        st = libFile.get_sigma(mt, at_energies=e)

    return e, st

def set_log_scale(ax, log_x, log_y):
    """
    Changes ax scale as desired

    Parameters
    ----------
    ax : matplotlib.ax
    log_x : bool
        True to set scale to log
    log_y : bool
        True to set scale to log

    """
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')


def plot_sampled_info(ace_file, h, zaid, mt, sample_df, sample_df_full_vals, zaid_2=None, mt_2=None, output_base='',
                      log_x=True, log_y=True, log_y_stddev=False, corr_rel_diff=False,
                      use_max_range_for_correlation_plot=True):
    """
    Plots diag of cov, a few xsecs, and full corr matrix of the sampled data.

    For mt=1018, only plots the first outgoing angle and the pdf

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
    log_x : bool
        True to set axis scale to log
    log_y : bool
        True to set axis scale to log
    log_y_stddev : bool
        True to set std dev plot to log y axis
    corr_rel_diff : bool
        True to output original corr matrix and abs value of the difference of sampled corr and original
    use_max_range_for_correlation_plot : bool
        Sets range to -1, 1 on correlation plot else uses the min/max of the actual data
    """
    # plot the cov
    xsec, corr = XsecSampler.load_zaid_mt(h, zaid, mt, zaid_2, mt_2)

    e_for_plot = list(xsec['e high'])
    e_for_plot.append(e_for_plot[-1] / 2)

    y_for_plot = list(xsec['s.d.(1)'] ** 2)
    y_for_plot.append(y_for_plot[-1])

    y2_for_plot = list(np.diag(np.cov(sample_df_full_vals)))
    y2_for_plot.append(y2_for_plot[-1])

    fig, ax = plt.subplots()
    ax.plot(e_for_plot, y_for_plot, drawstyle='steps-post', label='Diag(cov) ENDF')
    ax.plot(e_for_plot, y2_for_plot, drawstyle='steps-post', label='Diag(cov) Sampled')
    set_log_scale(ax, log_x, log_y)

    ax.legend()
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Covariance")
    plt.savefig("{2}{3}{0}_{1}_sampled_cov.png".format(zaid, mt, output_base, os.path.sep), bbox_inches='tight', dpi=450)

    # Plot some xsec
    e, st = get_mt_from_ace(ace_file, mt)

    fig, ax = plt.subplots()

    # ensure we don't try to plot too many samples
    num_xsec = 10

    # if less than num_xsec samples, plot all of them
    if num_xsec > sample_df.shape[1]:
        num_xsec = sample_df.shape[1]

    if mt == 1018:
        e = e[0]
        st = st[0]

    # plot the base x-sec too
    ax.plot(e, st, linestyle='-.', color='k')

    for i in range(num_xsec):
        sampled_xsec = map_groups_to_continuous(e, xsec['e high'], sample_df.iloc[:, i], min_e=xsec['e low'].min()) * st
        ax.plot(e, sampled_xsec, label=i)

    # plot the base again so it appears on top
    ax.plot(e, st, linestyle='-.', color='k')

    # to see if we should log, check only nonzero values
    # let it be ugly if zome zeros found?
    nonzero_st = st[np.nonzero(st)]
    if max(nonzero_st)/min(nonzero_st) > 1000:
        set_log_scale(ax, log_x, True)
    else:
        set_log_scale(ax, log_x, False)

    ax.legend(['Actual', 'Samples'])

    ax.set_xlabel('Energy (Ev)')
    ax.set_ylabel('Cross Section (b)')

    plt.savefig("{2}{3}{0}_{1}_few_sampled_xsec.png".format(zaid, mt, output_base, os.path.sep), bbox_inches='tight', dpi=450)

    # Plot the corr matrix
    # must flip the corr because it is in high energy to low energy form
    e_for_corr = list(xsec['e high'])

    # make sure the elow isn't 0 so if we log the data it doesn't break..
    if xsec['e low'].values[-1] == 0:
        elow_to_append = xsec['e low'].values[-2] / 2
    else:
        elow_to_append = xsec['e low'].values[-1]

    e_for_corr.append(elow_to_append)
    e_for_corr = e_for_corr[-1::-1]
    X, Y = np.meshgrid(e_for_corr, e_for_corr)

    fig, ax = plt.subplots(ncols=2, figsize=(8, 6), sharey=True)

    if corr_rel_diff:
        corr_for_plot2 = np.abs(np.corrcoef(sample_df_full_vals) - corr.values)
    else:
        corr_for_plot2 = np.corrcoef(sample_df_full_vals)
        corr_for_plot2[np.isnan(corr_for_plot2)] = 0

        # if the original corr was 0 on the diag, the sampled distro was not actually sampled so lets not plot that
        original_corr_zero_idx = np.diag(corr.values) == 0
        corr_for_plot2[:, original_corr_zero_idx] = 0
        corr_for_plot2[original_corr_zero_idx, :] = 0

    if use_max_range_for_correlation_plot:
        vmin = -1.0
        vmax = 1.0
    else:
        vmin = np.amax(corr.values)
        vmax = np.amin(corr.values)

    if np.amax(corr_for_plot2) > vmax:
        vmax = np.amax(corr_for_plot2)

    if np.amin(corr_for_plot2) < vmin:
        vmin = np.amin(corr_for_plot2)

    corr_flipped_for_plot = np.flipud(np.fliplr(corr.values))
    im = ax[0].pcolormesh(X, Y, corr_flipped_for_plot, vmin=vmin, vmax=vmax, cmap=cm_RdGr)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[0].xaxis.tick_top()
    ax[0].set_xlabel('Energy (eV)')
    ax[0].set_ylabel('Energy (eV)')
    ax[0].xaxis.set_label_position('top')

    ax[0].set_xlim([e_for_corr[0], e_for_corr[-1]])
    ax[0].set_ylim([e_for_corr[0], e_for_corr[-1]])

    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set(adjustable='box', aspect='equal')

    # second corr plot

    corr_flipped_for_plot = np.flipud(np.fliplr(corr_for_plot2))
    im = ax[1].pcolormesh(X, Y, corr_flipped_for_plot, vmin=vmin, vmax=vmax, cmap=cm_RdGr)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    ax[1].xaxis.tick_top()
    ax[1].set_xlabel('Energy (eV)')
    ax[1].xaxis.set_label_position('top')

    ax[1].set_xlim([e_for_corr[0], e_for_corr[-1]])
    ax[1].set_ylim([e_for_corr[0], e_for_corr[-1]])

    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set(adjustable='box', aspect='equal')

    plt.gca().invert_yaxis()

    fig.tight_layout()
    plt.savefig("{2}{3}{0}_{1}_sampled_corr_compare.png".format(zaid, mt, output_base, os.path.sep), dpi=450,
                bbox_inches='tight')

    plot_xsec(ace_file, h, zaid, mt, output_base, log_y_stddev=log_y_stddev)


def plot_xsec(ace_file, h, zaid, mt, output_base='./', pad_rel_y_decades=False, log_x=True, log_y=True, log_y_stddev=False):
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
    pad_rel_y_decades : bool
        Ensures at least two decades of standard deviation is plotted
    log_x : bool
        True to set axis scale to log
    log_y : bool
        True to set axis scale to log
    log_y_stddev : bool
        True to set stddev scale to log

    Returns
    -------

    """
    # plot the xsec + the std_dev
    xsec, corr = XsecSampler.load_zaid_mt(h, zaid, mt, zaid, mt)
    e, st = get_mt_from_ace(ace_file, mt)

    if mt == 1018:
        e = e[0]
        st = st[0]

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.3])
    ax = fig.add_subplot(gs[0])
    ax.plot(e, st)
    ax.grid(alpha=0.25)
    # turn off the x labels w/o turning off the grid x
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    ax.set_ylabel('Cross-Section (b)')

    set_log_scale(ax, log_x, log_y)

    # second plot (rel dev %)
    ax2 = fig.add_subplot(gs[1], sharex=ax)  # the above ax

    # check if lowest e is much higher than the plotted xsec so it looks okay
    if e[0] < xsec['e high'].values[-1]:
        e_for_plot = list(xsec['e high'])
        e_for_plot.append(e[0])

        y_for_plot = list(xsec['rel.s.d.(1)'] * 100)
        y_for_plot.append(y_for_plot[-1])
    else:
        # do not need to add anything
        e_for_plot = xsec['e high']
        y_for_plot = xsec['rel.s.d.(1)'] * 100

    ax2.plot(e_for_plot, y_for_plot, drawstyle='steps-post')

    # auto scale y log under some conditions
    if min(y_for_plot) > 0 and max(y_for_plot) / min(y_for_plot) > 100:
        log_y = True
    elif min(y_for_plot) == 0 and max(y_for_plot) > 100:
        log_y = True

    set_log_scale(ax2, log_x, log_y)

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

    if log_y_stddev:
        ax2.set_yscale('log')

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    plt.savefig("{2}{3}{0}_{1}_base_xsec_with_std.png".format(zaid, mt, output_base, os.path.sep), bbox_inches='tight', dpi=450)


def write_sampled_data(h, ace_file, zaid, mts, sample_dfs_rel, output_formatter='xsec_sampled_{0}', zaid_2=None, mt_2=None):
    """
    Write sampled data to files + the group-wise sampled data to a csv

    :param h: the cov h5 handle
    :param ace_file:
    :param sample_df_rel:
    :param output_formatter: str, 'some_text_{0}_formatter_to_add_number_to'
    :return:
    """

    # sample_df contains relative values (sampled / mean) which are then multiplied by the actual continuous xsec and written to an ace file

    # iter over # of samples, each processor does 1 whole ace file no matter how many mts were sampled
    for sample_number, _ in sample_dfs_rel[0].iteritems():
        if sample_number % size != rank: continue

        # load the base ace file to get a fresh copy
        ae = AceIO.AceEditor(ace_file)

        # iter over mts, with the idea that the mts were given in the same order as the sample_dfs
        for mt_idx, mt in enumerate(mts):
            col = sample_dfs_rel[mt_idx][sample_number]
            # load the cov to get the mapping
            xsec, corr = XsecSampler.load_zaid_mt(h, zaid, mt, zaid_2, mt_2)

            e = ae.get_energy(mt)
            original_sigma = ae.get_sigma(mt)

            if mt == 1018:
                # assumes all incident n / e_out have the SAME pdf relative samples
                sampled_xsec = []
                for pdf_for_e_out, e_to_map_on in zip(original_sigma, e):
                    sampled_xsec.append(map_groups_to_continuous(e_to_map_on, xsec['e high'], col,
                                                                 min_e=xsec['e low'].min()) * pdf_for_e_out)
            else:
                # in case we get a 0 mean set it to 1.0. The sample_df sets rel_dev
                original_sigma[original_sigma == 0] = 1

                sampled_xsec = map_groups_to_continuous(e, xsec['e high'], col,
                                                        min_e=xsec['e low'].min()) * original_sigma

            ae.set_sigma(mt, sampled_xsec)
            ae.apply_sum_rules()

        w = AceIO.WriteAce(ace_file)
        base_ace = AceIO.AceEditor(ace_file)
        # loop low MT to high because that's how the ace file is written. This helps avoid cases where a redundant MT
        # might replaces multiple MTs at once which may or may not be what we wanted

        for mt_adjusted in sorted(list(ae.adjusted_mts)):
            try:
                if mt_adjusted == 452:
                    max_replaces = 9999
                else:
                    max_replaces = 1

                if mt_adjusted == 1018:
                    _, _, base_fission_chi_prompt_energy_p, base_fission_chi_prompt_energy_c = base_ace.get_chi_distro()
                    _, _, fission_chi_prompt_energy_p, fission_chi_prompt_energy_c = ae.get_chi_distro()

                    for _base, _sampled in zip(base_fission_chi_prompt_energy_p, fission_chi_prompt_energy_p):
                        # prob/MeV converted back to per eV
                        w.replace_array(_base * 1e6, _sampled * 1e6, max_replaces=1)

                    for _base, _sampled in zip(base_fission_chi_prompt_energy_c, fission_chi_prompt_energy_c):
                        w.replace_array(_base, _sampled, max_replaces=1)
                else:
                    w.replace_array(base_ace.get_sigma(mt_adjusted), ae.get_sigma(mt_adjusted), max_replaces)
            except ValueError:
                print("MT {0} adjusted but was not present on original ace, perhaps it was redundant".format(mt_adjusted))

        w.write_ace(output_formatter.format(sample_number))

        del ae

    if rank == 0:
        # add in e groups then print all relative data and the base xsec
        for mt, sample_df_rel in zip(mts, sample_dfs_rel):
            xsec, corr = XsecSampler.load_zaid_mt(h, zaid, mt, zaid_2, mt_2)
            sample_df_rel.insert(0, 'e low', xsec['e low'])
            sample_df_rel.insert(1, 'e high', xsec['e high'])
            sample_df_rel.insert(2, 'xsec', xsec['x-sec(1)'])
            sample_df_rel.insert(3, 'rel std', xsec['rel.s.d.(1)'])
            sample_df_rel.to_csv('{0}_samples.csv'.format(output_formatter.format(mt)))

def create_argparser():
    parser = argparse.ArgumentParser(
        description="Generate random samples of ACE data for use in SA/UQ")
    parser.add_argument('base_ace', help="The base ACE file to sample from")
    parser.add_argument('cov_store', help="The ASAPy covariance store to use")
    parser.add_argument('num_samples', help="Number of samples to draw", type=int)
    parser.add_argument('mts', help="The reaction MT number to sample", type=int, nargs='+')

    parser.add_argument('-mpiproc', type=int, help="Number of mpiprocs to use", default=1)
    parser.add_argument('-num_oversamples',
                        help="Make this many samples but only keep num_samples. Helps if negative samples are being drawn due to large uncertainties in small numbers",
                        type=int, default=-1)
    parser.add_argument('--make_plots', action='store_true',
                        help="Option to create sampled cov plot, corr plot, xsec + uncertainties, and a few sampled xsec")
    parser.add_argument('--writepbs', action='store_true', help="Creates a pbs file to run this function")
    parser.add_argument('--waitforjob', help="Job number to wait for until this job runs")
    parser.add_argument('--subpbs', action='store_true', help="Runs the created pbs file")
    parser.add_argument('-distribution',
                        help="Choose between norm. lognorm, uncorrelated, loguncorrelated, or uniform sampling", default='normal')

    return parser

if __name__ == "__main__":
    import os
    from coupleorigen import qsub_helper
    import argparse

    parser = create_argparser()
    args = parser.parse_args()

    sample_choices = ['norm', 'lognorm', 'uncorrelated', 'uniform', 'loguncorrelated']
    if args.distribution.lower() not in sample_choices:
        raise Exception(
            "Unknown sample distribution {0}, please choose from {1}".format(args.distribution, sample_choices))

    if args.writepbs:
        pbs_args = {}
        pbs_args['depends_on'] = args.waitforjob

        if args.make_plots:
            make_plots = '--make_plots'
        else:
            make_plots = ''

        if args.mpiproc > 1:
            mpi_cmd = 'mpiexec -n {0} '.format(args.mpiproc)
        else:
            mpi_cmd = ''

        mts = [str(i) for i in args.mts]
        mts = ' '.join(mts)

        python_to_run = mpi_cmd + 'python ' + os.path.abspath(__file__)


        qsub_helper.qsub_helper('sample_xsec.sh', [python_to_run], [
            "{0} {1} {5} {2} -distribution {3} -num_oversamples {4} {6}".format(args.base_ace, args.cov_store,
                                                                                             mts, args.distribution,
                                                                                             args.num_oversamples,
                                                                                             args.num_samples, make_plots)],
                                pbs_args=pbs_args, mpiprocs=args.mpiproc)

        if args.subpbs:
            os.system('qsub sample_xsec.sh')

    else:
        # do all this on all procs to avoid writing comm code..
        ace_file = args.base_ace
        ace_data = AceIO.AceEditor(ace_file)
        zaid = str(ace_data.table.atomic_number) + str(ace_data.table.mass_number)
        atomic_symbol = ace_data.table.atomic_symbol

        store_name = args.cov_store
        num_samples_to_take = args.num_samples

        if args.num_oversamples == -1:
            num_samples_to_make = num_samples_to_take
        else:
            num_samples_to_make = args.num_oversamples

        if num_samples_to_make < num_samples_to_take:
            raise Exception("Cannot make more samples {0} than taking {1}".format(num_samples_to_make, num_samples_to_take))

        output_base = './'
        # if we ever take a user output_base, we might want to make the folder to
        # os.makedirs(output_base, exist_ok=True)
        sample_dfs = []
        sample_dfs_full = []

        with pd.HDFStore(os.path.expanduser(store_name), 'r') as h:
            if rank == 0:
                for mt in args.mts:
                    sample_df, sample_df_full = sample_xsec(h, mt, zaid, num_samples_to_take,
                                                            num_samples_to_make=num_samples_to_make,
                                                            sample_type=args.distribution, raise_on_bad_sample=False,
                                                            remove_neg=True)

                    if args.make_plots:
                        plot_sampled_info(ace_file, h, zaid, mt, sample_df, sample_df_full, output_base=output_base,
                                          log_y=True, log_y_stddev=False)

                    sample_dfs.append(sample_df)
            else:
                sample_dfs = None

            sample_dfs = comm.bcast(sample_dfs, root=0)
            output_formatter = '{0}{1}'.format(atomic_symbol, ace_data.table.mass_number)  # mt is written by the output
            write_sampled_data(h, ace_file, zaid, args.mts, sample_dfs, output_formatter=output_base + output_formatter + '_{0}')
