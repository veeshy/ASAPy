import unittest
from ASAPy import CovManipulation as cm
import numpy as np

"""
These tests are statistical in nature and will filly in a val grind
"""

class Test_update_mcnp_input(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.means = np.array(np.ones(25) * 20)
        cls.desired_corr = np.diag([1] * 25) + np.diag([-0.5] * 24, 1) + np.diag([-0.5] * 24, -1)
        cls.std_dev = np.ones(25) * 0.05 * 20
        cls.samples = 500

    def test_correlated_norm_lhs(self):
        """
        Draw 500 samples from a 238-len vector with random means
        Returns
        -------

        """

        dependent_samples = cm.lhs_normal_sample_corr(self.means, self.std_dev, self.desired_corr, self.samples)

        corr = np.corrcoef(dependent_samples.T)

        # want the found correlations to be within 10%
        min_c = np.diagonal(corr, offset=1).min()
        max_c = np.diagonal(corr, offset=1).max()

        if min_c < -0.6:
            self.fail("Min corr created too small: {0}".format(min_c))

        if max_c > -0.4:
            self.fail("Max corr created too small: {0}".format(max_c))

    def test_correlated_norm(self):
        """
        Draw 500 samples from a 238-len vector with random means
        Returns
        -------

        """
        cov = cm.correlation_to_cov(self.std_dev, self.desired_corr)
        import matplotlib.pyplot as plt

        dependent_samples = cm.normal_sample_corr(self.means, cov, self.samples)

        corr = np.corrcoef(dependent_samples.T)

        # want the found correlations to be within 10%
        min_c = np.diagonal(corr, offset=1).min()
        max_c = np.diagonal(corr, offset=1).max()

        if min_c < -0.6:
            self.fail("Min corr created too small: {0}".format(min_c))

        if max_c > -0.4:
            self.fail("Max corr created too small: {0}".format(max_c))

    def test_cov_corr(self):

        some_matrix = np.random.rand(10,10)
        corr = np.corrcoef(some_matrix)
        cov = np.cov(some_matrix)

        cov_calc = cm.correlation_to_cov(np.diag(cov)**0.5, corr)
        corr_calc = cm.cov_to_correlation(cov)

        np.testing.assert_almost_equal(cov, cov_calc)
        np.testing.assert_almost_equal(corr, corr_calc)


if __name__ == '__main__':
    unittest.main()
