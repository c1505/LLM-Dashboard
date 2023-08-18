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
        self.assertEqual(len(data), 992)
        
if __name__ == '__main__':
    unittest.main()