"""
Rubric for creating new correlation and stddev df's for storing in ASAPy format

The ASAPy data format comes from how SCALE cov is outputted
"""

import pandas as pd

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

def check_correlation_df(df):
    assert len(df.columns) == len(df.index), "Non square correlation matrix"
    assert df.columns.values == df.index.values, "Group labels not equal in correlation matrix"

def add_corr_to_store(store, df, mat1, mt1, mat2, mt2):
    """
    Adds df to the store with the ASAPy key format
    Parameters
    ----------
    store : HDF5store
        The store to add to
    df
        The correlation format

    Returns
    -------

    """

    check_correlation_df(df)
    store.put(df_key_corr_formatter.format(mat1, mt1, mat2, mt2), df)

def check_stddev_df(df):
    assert df.values[0, 0] < df.values[1, 0], "Energy bins should be in decreasing order"

def add_stddev_to_store(store, df, mat1, mt1, mat2, mt2):
    """
    Adds df to the store with the ASAPy key format
    Parameters
    ----------
    store : HDF5store
        The store to add to
    df
        The correlation format

    Returns
    -------

    """

    check_stddev_df(df)
    store.put(df_key_stddev_formatter.format(mat1, mt1, mat2, mt2), df)
