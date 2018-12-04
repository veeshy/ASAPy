from unittest import TestCase
from ASAPy import XsecSampler
import unittest
import pandas as pd
import numpy as np


class TestMapGroups(TestCase):
    def test_map_groups_to_continuous(self):
        """
        Expect a certain mapping. Points are mapped from e_sigma to high_e_bins based on where e_sigma
        would be if it was inserted into high_e_bins. This is to figure out what energy group # the e sigma lies

        Returns
        -------

        """
        e_sigma = np.array([1e-8, 1e-3, 1.0, 20])  # MeV
        high_e_bins = pd.Series([25, 2.0, 1e-1, 1e-4, 1e-5]) * 1e6  # eV
        multi_group_val = pd.Series([1, 2, 3, 4, 5])
        max_e = 15e6  # eV
        min_e = 1.0  # eV
        mapped_values = XsecSampler.map_groups_to_continuous(e_sigma, high_e_bins, multi_group_val, max_e, min_e)

        self.assertListEqual(list(mapped_values), [5, 5, 4, 1])


if __name__ == "__main__":
    unittest.main()