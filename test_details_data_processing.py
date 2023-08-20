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
        DetailsDataProcessor.download_file('https://www.google.com', 'test.html')
        self.assertTrue(os.path.exists('test.html'))
        os.remove('test.html')

    def test_generate_url(self):
        results_file_path = "64bits/LexPodLM-13B/results_2023-07-25T13:41:51.227672.json"
        expected_url = 'https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/64bits/LexPodLM-13B/details_harness%7ChendrycksTest-moral_scenarios%7C5_2023-07-25T13%3A41%3A51.227672.json'


        constructed_url = self.processor.generate_url(results_file_path)
        self.assertEqual(expected_url, constructed_url)

    def test_pipeline(self):
        df = self.processor.pipeline()
        print(100 * "****")
        print(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_find_files(self):
        directory = 'results'
        pattern = '*moral*.json'
        files = self.processor._find_files(directory, pattern)
        breakpoint()
        print(files)
        self.assertIsInstance(files, list)

        
if __name__ == '__main__':
    unittest.main()