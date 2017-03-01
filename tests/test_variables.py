"""Test of lax/variables.py"""
import unittest
from collections import OrderedDict
from lax import variables


class VariablesTestCase(unittest.TestCase):
    """Test case for lax/variables.py
    """

    def test_type(self):
        """Test type of variable getter"""
        self.assertIsInstance(variables.get_variables(),
                              OrderedDict)


if __name__ == '__main__':
    unittest.main()
