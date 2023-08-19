import unittest
from details_data_processor import DetailsDataProcessor
import pandas as pd

class TestDetailsDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DetailsDataProcessor()

    # check that the result is a pandas dataframe
    def test_process_data(self):
        pass
        # data = self.processor.data
        # self.assertIsInstance(data, pd.DataFrame)

        
if __name__ == '__main__':
    unittest.main()