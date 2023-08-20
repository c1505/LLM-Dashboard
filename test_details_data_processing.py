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
        DetailsDataProcessor.download_file('https://www.google.com', 'test_file_please_remove')
        self.assertTrue(os.path.exists('test.html'))
        os.remove('test.html')


    # there is a file path that results in a 404 error.
    # create a test replicate that error
    def test_download_file_404(self):
        file_path_with_error = 'results/shaohang/Sparse0.5_OPT-1.3/results_2023-07-19T19:10:31.005235.json'
        url = self.processor.build_url(file_path_with_error)
        DetailsDataProcessor.download_file(url, 'test_file_please_remove')


    def test_build_url(self):
        test_cases = [
            ('results/64bits/LexPodLM-13B/results_2023-07-25T13:41:51.227672.json',
            'https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/64bits/LexPodLM-13B/details_harness%7ChendrycksTest-moral_scenarios%7C5_2023-07-25T13%3A41%3A51.227672.json'),
            ('results/AlpinDale/pygmalion-instruct/results_2023-08-17T11:20:15.687659.json',
            'https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/AlpinDale/pygmalion-instruct/details_harness%7ChendrycksTest-moral_scenarios%7C5_2023-08-17T11%3A20%3A15.687659.json')
        ]
    
        for file_path, expected in test_cases:
            assert self.processor.build_url(file_path) == expected, f"Test failed for file_path: {file_path}"

    def test_pipeline(self):
        df = self.processor.pipeline()
        print(100 * "****")
        print(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_find_files(self):
        directory = 'results'
        pattern = 'results*.json'
        files = self.processor._find_files(directory, pattern)
        # breakpoint()
        # print(files)
        self.assertIsInstance(files, list)

        
if __name__ == '__main__':
    unittest.main()