import pandas as pd
import os
import fnmatch
import json
import re
import numpy as np
import requests

class DetailsDataProcessor:
    # Download 
    #url example https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/64bits/LexPodLM-13B/details_harness%7ChendrycksTest-moral_scenarios%7C5_2023-07-25T13%3A41%3A51.227672.json
    
    def __init__(self, directory='results', pattern='results*.json'):
        self.directory = directory
        self.pattern = pattern
        # self.data = self.process_data()
        # self.ranked_data = self.rank_data()



    # download a file from a single url and save it to a local directory
    @staticmethod
    def download_file(url, filename):
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)

    @staticmethod
    def single_file_pipeline(url, filename):
        DetailsDataProcessor.download_file(url, filename)
        # read file
        with open(filename) as f:
            data = json.load(f)
        # convert to dataframe
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def generate_url(file_path):
        base_url = 'https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/'


        organization = '64bits'
        model = 'LexPodLM-13B'
        filename = '_2023-07-25T13%3A41%3A51.227672.json'
        # extract organization, model, and filename from file_path instead of hardcoding
        # filename = file_path.split('/')[-1]



        other_chunk = 'details_harness%7ChendrycksTest-moral_scenarios%7C5'
        constructed_url = base_url + organization + '/' + model + '/' + other_chunk + filename
        return constructed_url


    def _find_files(self, directory, pattern):
        matching_files = []  # List to hold matching filenames
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    matching_files.append(filename)  # Append the matching filename to the list
        return matching_files  # Return the list of matching filenames

    
    def pipeline(self):
        dataframes = []
        file_paths = self._find_files(self.directory, self.pattern)
        for file_path in file_paths:
            print(file_path)
            url = self.generate_url(file_path)
            file_path = file_path.split('/')[-1]
            df = self.single_file_pipeline(url, file_path)
            dataframes.append(df)
        return dataframes
