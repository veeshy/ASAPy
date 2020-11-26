from unittest import TestCase
from ASAPy import EndfToCov, njoy
import unittest
import shutil
import os
import difflib


class TestReadBoxer2Matrix(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.read_boxer_out_matrix = EndfToCov.read_boxer_out_matrix('./test_data/boxer2mat_out.txt')

    def testFindBlockNums(self):
        block_nums = self.read_boxer_out_matrix.block_line_nums
        self.assertListEqual(block_nums, [4, 39, 74, 109])

    def testReadBlockLinesToArray(self):
        values = self.read_boxer_out_matrix._block_lines_to_array(self.read_boxer_out_matrix.block_line_nums[1]+1, self.read_boxer_out_matrix.block_line_nums[2])
        # check an arbitrary known entry
        self.assertEqual(values[87], 34.96)
        # check len
        self.assertEqual(len(values), 238)

    def testCov(self):
        _,_,std_dev,cov = self.read_boxer_out_matrix.get_block_data()
        self.assertEqual(len(cov), 238)

class TestRunCoverChain(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = './test_data/run_cover_chain_test_out/'

    def testChain(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        EndfToCov.run_cover_chain("./test_data/n_0125_1-H-1.dat", [2, 102], [300, 2400], output_dir=self.test_dir,
                                  user_flux_weight_vals=6, cov_energy_groups=njoy.energy_groups_238,
                                  njoy_exec='/Users/veeshy/projects/NJOY2016/bin/njoy',
                                  boxer_exec='/Users/veeshy/projects/ASAPy/boxer2mat/boxer2mat')

        # check each file against gold

        files = ["covr_2400.txt_102_matrix.txt", "testing_chain.txtboxer.txt", "covr_2400.txt_2_matrix.txt",
                 "covr_300.txt_102_matrix.txt", "covr_300.txt_2_matrix.txt", "viewr_2400.eps", "covr_2400.txt",
                 "viewr_300.eps", "covr_300.txt", "testing_chain.txt"]

        gold_dir = "./test_data/gold_njoy_boxer_chain_test_out/"

        for file in files:
            with open(os.path.join(gold_dir, file)) as g:
                with open(os.path.join(self.test_dir, file)) as t:
                    gold_lines = g.readlines()
                    test_lines = t.readlines()
                    for gold_line, test_line in zip(gold_lines, test_lines):
                        self.assertMultiLineEqual(gold_line, test_line, msg="File: {0}".format(file))

            os.remove(os.path.join(self.test_dir, file))

    def tearDown(self):
        try:
            os.rmdir(self.test_dir)
        except:
            print("Could not remove ./test_data/run_cover_chain_test_out/ directory because it is not empty.")

    def assertMultiLineEqual(self, first, second, msg=None):
        """Assert that two multi-line strings are equal.

        If they aren't, show a nice diff.

        """
        self.assertTrue(isinstance(first, str),
                        'First argument is not a string')
        self.assertTrue(isinstance(second, str),
                        'Second argument is not a string')

        if first != second:
            message = ''.join(difflib.ndiff(first.splitlines(True),
                                            second.splitlines(True)))
            if msg:
                message += " : " + msg
            self.fail("Multi-line strings are unequal:\n" + message)


if __name__ == "__main__":
    unittest.main()