import unittest
from result_data_processor import ResultDataProcessor
import pandas as pd

class TestResultDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = ResultDataProcessor()

    # check that the result is a pandas dataframe
    def test_process_data(self):
        data = self.processor.data
        self.assertIsInstance(data, pd.DataFrame)
        
    # check that pandas dataframe has the right columns
    def test_columns(self):
        data = self.processor.data
        self.assertIn('Parameters', data.columns)
        self.assertIn('MMLU_average', data.columns)
        # check number of columns
        self.assertEqual(len(data.columns), 63)

    # check that the number of rows is correct
    def test_rows(self):
        data = self.processor.data
        self.assertEqual(len(data), 998)

    # check that mc1 column exists
    def test_mc1(self):
        data = self.processor.data
        self.assertIn('harness|truthfulqa:mc1', data.columns)

    # test that a column that contains truthfulqa:mc does not exist
    def test_truthfulqa_mc(self):
        data = self.processor.data
        self.assertNotIn('truthfulqa:mc', data.columns)

    # check for extreme outliers in mc1 column
    def test_mc1_outliers(self):
        data = self.processor.data
        mc1 = data['harness|truthfulqa:mc1']
        self.assertLess(mc1.max(), 1.0)
        self.assertGreater(mc1.min(), 0.0)
        

    # test that a column named organization exists
    def test_organization(self):
        data = self.processor.data
        self.assertIn('organization', data.columns)

if __name__ == '__main__':
    unittest.main()