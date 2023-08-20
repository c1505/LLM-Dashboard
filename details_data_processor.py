import pandas as pd
import os
import fnmatch
import json
import re
import numpy as np
import requests
from urllib.parse import quote

class DetailsDataProcessor:
    # Download 
    #url example https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/64bits/LexPodLM-13B/details_harness%7ChendrycksTest-moral_scenarios%7C5_2023-07-25T13%3A41%3A51.227672.json
    
    def __init__(self, directory='results', pattern='results*.json'):
        self.directory = directory
        self.pattern = pattern
        # self.data = self.process_data()
        # self.ranked_data = self.rank_data()

    def _find_files(self, directory='results', pattern='results*.json'):
        matching_files = []  # List to hold matching filenames
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    matching_files.append(filename)  # Append the matching filename to the list
        return matching_files  # Return the list of matching filenames

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
    def build_url(file_path):
        segments = file_path.split('/')
        bits = segments[1]
        model_name = segments[2]
        timestamp = segments[3].split('_')[1]
        
        url = f'https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/{bits}/{model_name}/details_harness%7ChendrycksTest-moral_scenarios%7C5_{quote(timestamp, safe="")}'
        print(url)
        return url
    
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
