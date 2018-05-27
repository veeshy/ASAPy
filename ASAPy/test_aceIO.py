from unittest import TestCase
import unittest
from ASAPy import AceIO

class TestAceEditor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ace = AceIO.AceEditor('../test_data/92235.710nc')

    def test_get_nu_distro(self):
        [e, v] = self.ace.get_nu_distro()
        self.assertAlmostEqual(e[4], 0.0001)
        self.assertAlmostEqual(v[4], 2.4338)

    def test_get_sigma(self):
        v = self.ace.get_sigma(2)
        self.assertAlmostEqual(v[99], 18.65279)

    def test_set_sigma_wrong_len(self):
        self.assertRaises(IndexError, self.ace.set_sigma, *[2, [5, 10]])

    def test_set_sigma_(self):
        self.ace.set_sigma(2, self.ace.get_sigma(2)*2)

        v = self.ace.get_sigma(2)
        self.assertAlmostEqual(v[99], 18.65279*2)
        self.assertEqual(self.ace.adjusted_mts, {2})

    def test_non_existant_sigma(self):
        self.assertRaises(ValueError, self.ace.get_sigma, 3)

    def test_set_sum(self):
        #TODO need to find a good test for this
        pass
        #self.ace.set_sigma(102, self.ace.get_sigma(2)*2)
        #self.ace.apply_sum_rules()


    def test__check_if_mts_present(self):
        """
        Make sure all MTs are found correctly
        """
        should_be_in_mts = [2, 18, 102]
        mts_found = self.ace._check_if_mts_present(should_be_in_mts)
        self.assertEqual(mts_found, should_be_in_mts)

    def test__check_if_some_mts_present(self):
        """
        Make sure if some MT in the list of MTs is not there, it is removed
        """
        should_be_in_mts = [2, 18, 102]
        should_not_be_in_mts = [1000]
        mts_found = self.ace._check_if_mts_present(should_be_in_mts + should_not_be_in_mts)
        self.assertEqual(mts_found, should_be_in_mts)

if __name__ == '__main__':
    unittest.main()
