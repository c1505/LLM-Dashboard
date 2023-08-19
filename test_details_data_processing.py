import unittest
from details_data_processor import DetailsDataProcessor
import pandas as pd
import requests
import os

class TestDetailsDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DetailsDataProcessor()

    # check that the result is a pandas dataframe
    def test_process_data(self):
        pass
        # data = self.processor.data
        # self.assertIsInstance(data, pd.DataFrame)

    def test_download_file(self):
        DetailsDataProcessor._download_file('https://www.google.com', 'test.html')
        self.assertTrue(os.path.exists('test.html'))
        os.remove('test.html')

        
if __name__ == '__main__':
    unittest.main()