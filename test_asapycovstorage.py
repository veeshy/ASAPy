from unittest import TestCase
from ASAPy import AsapyCovStorage

class TestCreate_stddev_df(TestCase):
    def test_construct(self):
        df = AsapyCovStorage.create_stddev_df(5)
        self.assertListEqual(list(df.columns), ['groups', 'e high', 'x-sec(1)', 'rel.s.d.(1)', 's.d.(1)'])
        self.assertListEqual(list(df.index), list(range(5)))


class TestCreate_corr_df(TestCase):
    def test_construct(self):
        df = AsapyCovStorage.create_corr_df(5)
        self.assertListEqual(list(df.columns), list(range(5)))
        self.assertListEqual(list(df.index), list(range(5)))
