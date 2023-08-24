import unittest
import os

class TestPaths(unittest.TestCase):
    def test_path_exists(self):
        # test that the path results exists
        self.assertTrue(os.path.exists('results'))

    def test_results_directory_is_not_empty(self):
        # test that the results directory is not empty
        self.assertGreater(len(os.listdir('results')), 0)

    def test_results_contain_json_files(self):
        # test that the results director contains json files in the sub directores
        # get a list of all the subdirectories
        subdirectories = [x[0] for x in os.walk('results')]
        # check if the subdirectories contain json files. only check one subdirectory
        subdirectory = subdirectories[1]
        self.assertGreater(len(os.listdir(subdirectory)), 0)