from unittest import TestCase
from ASAPy import EndfToCov
import unittest

class TestReadBoxer2Matrix(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.read_boxer_out_matrix = EndfToCov.read_boxer_out_matrix('../test_data/boxer2mat_out.txt')

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

if __name__ == "__main__":
    unittest.main()