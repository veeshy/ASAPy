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
        cls.samples = 1000

    def test_correlated_norm_lhs_norm(self):
        """
        Draw samples from a 238-len vector with random means
        Returns
        -------

        """
        dependent_samples = cm.sample_with_corr(self.means, self.std_dev, self.desired_corr, self.samples)
        corr = np.corrcoef(dependent_samples.T)

        # divide by zero likely, it's okay though will be removed and max taken
        max_err = np.abs(np.divide(self.desired_corr - corr, self.desired_corr))
        max_err = np.max(max_err[np.isfinite(max_err)])

        if max_err > 0.03:
            self.fail(
                "Sampled distribution correlation does not agree within 3% of the desired correlation. Largest sampling err: {0}%".format(
                    max_err * 100))

    def test_correlated_norm_lhs_lognorm(self):
        """
        Draw samples from a 238-len vector with random means
        Returns
        -------

        """
        dependent_samples = cm.sample_with_corr(self.means, self.std_dev, self.desired_corr, self.samples, distro='lognorm')
        corr = np.corrcoef(dependent_samples.T)

        # divide by zero likely, it's okay though will be removed and max taken
        max_err = np.abs(np.divide(self.desired_corr - corr, self.desired_corr))
        max_err = np.max(max_err[np.isfinite(max_err)])

        if max_err > 0.03:
            self.fail(
                "Sampled distribution correlation does not agree within 3% of the desired correlation. Largest sampling err: {0}%".format(
                    max_err * 100))

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

    def test_gmw_cholesky(self):
        """
        Test gmw cholesky reconstruction for a PD case where the decomposition should be exact

        """

        desired_corr = np.array([[1.,     0.3309, 0.3121],
                                 [0.3309, 1.,     0.4475],
                                 [0.3121, 0.4475, 1.]])

        P, L, e = cm.gmw_cholesky(desired_corr)
        C = np.dot(P, L)
        gmw_cholesky_reconstructed = np.dot(C, C.T)

        np.testing.assert_allclose(gmw_cholesky_reconstructed, desired_corr)

    def test_gmw_cholesky_non_pd(self):
        """
        Tests gmw cholesky for a case where the matrix is not PD

        """

        desired_corr = np.array([[1.,     0.3309, 1.0],
                                 [0.3309, 1.,     0.4475],
                                 [1.0, 0.4475, 1.]])

        # P = np.linalg.cholesky(desired_corr) would fail due to not PD

        P, L, e = cm.gmw_cholesky(desired_corr)
        C = np.dot(P, L)
        gmw_cholesky_reconstructed = np.dot(C, C.T) - np.diag(np.dot(P, e))

        np.testing.assert_allclose(gmw_cholesky_reconstructed, desired_corr)

if __name__ == '__main__':
    unittest.main()
