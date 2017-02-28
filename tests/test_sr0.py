#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_lax
----------------------------------

Tests for `lax` module.
"""


import pandas as pd


from lax.lichens import sciencerun0




import unittest


class SR0Testcase(unittest.TestCase):
    def test_low_energy(self):
        df = pd.read_hdf('/Users/tunnell/Work/XENON/Analysis/Discrimination/discrimination.h5', 'table')
        sciencerun0.LowEnergy().process(df)

        #self.assertIsInstance(variables.get_variables(),
        #                      OrderedDict)



if __name__ == '__main__':
    unittest.main()
