"""
Rubric for creating new correlation and stddev df's for storing in ASAPy format
"""

import pandas as pd


# the ASAPy data format comes from how SCALE cov is outputted

# store corr and std_dev seperately
df_key_stddev_formatter = '{mat1}/{mt1}/{mat2}/{mt2}/std_dev'
df_key_corr_formatter = '{mat1}/{mt1}/{mat2}/{mt2}/corr'


# corr df format:
def _df_corr_index(n_groups):
    """
    Creates the correlation df index list which is based on the group #
    Parameters
    ----------
    n_groups

    Returns
    -------

    """
    return pd.Index(list(range(n_groups)))


def _df_corr_columns(n_groups):
    """
    Creates the correlation df column list which is based on the group #

    Parameters
    ----------
    n_groups

    Returns
    -------
    list

    """
    return list(range(n_groups))


def _df_stddev_columns():
    """
    The column labels for the ASAPy std dev df

    Returns
    -------
    list
    """
    return ['groups', 'e high', 'x-sec(1)', 'rel.s.d.(1)', 's.d.(1)']


def _df_stddev_index(n_groups):
    """
    The index labels or the ASAPy std dev df (energy group #)
    Parameters
    ----------
    n_groups

    Returns
    -------
    pd.Index

    """
    return pd.Index(list(range(n_groups)), name='groups')


def create_stddev_df(n_groups):
    df = pd.DataFrame(index=_df_stddev_index(n_groups), columns=_df_stddev_columns())
    return df


def create_corr_df(n_groups):
    df = pd.DataFrame(index=_df_corr_index(n_groups), columns=_df_corr_columns(n_groups))
    return df



