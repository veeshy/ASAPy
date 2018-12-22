from unittest import TestCase
import unittest
import filecmp
import os

from ASAPy import AceIO

class TestAceEditor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ace = AceIO.AceEditor('./test_data/92235.710nc')

    def test_write_endf_with_sum_rule(self):
        """
        Fake an adjusted xsec which should make mt rules go, write the data and it
        should be exactly as the input since nothing actually changed
        """
        ace = AceIO.AceEditor('./test_data/92235.710nc')
        ace.adjusted_mts.add(102)
        ace.apply_sum_rules()

        w = AceIO.WriteAce('./test_data/92235.710nc')
        base_ace = AceIO.AceEditor('./test_data/92235.710nc')
        for mt_adjusted in ace.adjusted_mts:
            try:
                w.replace_array(base_ace.get_sigma(mt_adjusted), base_ace.get_sigma(mt_adjusted))
            except ValueError:
                print("MT {0} adjusted but was not present on original ace, perhaps it was redundant".format(mt_adjusted))

        w.write_ace("./test_data/test_output_u235_102_no_changes")

        if not filecmp.cmp("./test_data/test_output_u235_102_no_changes", './test_data/92235.710nc'):
            self.fail("Created ace file not the same as original even though nothing changed.")
        else:
            os.remove("./test_data/test_output_u235_102_no_changes")


    def test_get_nu_distro(self):
        [e, v] = self.ace.get_nu_distro()
        self.assertAlmostEqual(e[4], 100.0)
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
        self.assertRaises(KeyError, self.ace.get_sigma, -1)

    def test_set_sum(self):
        #TODO need to find a good test for this
        pass
        #self.ace.set_sigma(102, self.ace.get_sigma(2)*2)
        #self.ace.apply_sum_rules()

    def test_get_nu(self):
        fission_chi_prompt_energy, fission_chi_prompt_energy_out, fission_chi_prompt_energy_p, fission_chi_prompt_energy_c = self.ace.get_chi_distro()
        self.assertAlmostEqual(fission_chi_prompt_energy_p[4][25], 3.368962e-09)

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
