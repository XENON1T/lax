import unittest
from collections import OrderedDict
from lax import variables

class VariablesTestCase(unittest.TestCase):
    def test_type(self):
        self.assertIsInstance(variables.get_variables(),
                              OrderedDict)

    def test_verbose(self):
        self.assertIsInstance(variables.get_variables(True),
                              OrderedDict)


        self.assertGreater(len(variables.get_variables(True)),
                           len(variables.get_variables(False)))

if __name__ == '__main__':
    unittest.main()
