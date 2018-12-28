from unittest import TestCase
import unittest
from ASAPy import AsapyCovStorage

class TestCreate_stddev_df(TestCase):
    def test_construct(self):
        df = AsapyCovStorage.create_stddev_df(5)
        self.assertListEqual(list(df.columns),
                             ['e low', 'e high', 'x-sec(1)', 'x-sec(2)', 'rel.s.d.(1)', 'rel.s.d(2)', 's.d.(1)',
                              's.d(2)'])
        self.assertListEqual(list(df.index), list(range(1, 6)))


class TestCreate_corr_df(TestCase):
    def test_construct(self):
        df = AsapyCovStorage.create_corr_df(5)
        self.assertListEqual(list(df.columns), list(range(1, 6)))
        self.assertListEqual(list(df.index), list(range(1, 6)))

if __name__ == "__main__":
    unittest.main()