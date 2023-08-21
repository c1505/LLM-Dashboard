import pandas as pd
import os
import fnmatch
import json
import re
import numpy as np
import requests
from urllib.parse import quote
from datetime import datetime
import uuid



class DetailsDataProcessor:
    # Download 
    #url example https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/64bits/LexPodLM-13B/details_harness%7ChendrycksTest-moral_scenarios%7C5_2023-07-25T13%3A41%3A51.227672.json
    
    # def __init__(self, directory='results', pattern='results*.json'):
    #     self.directory = directory
    #     self.pattern = pattern

    def __init__(self, directory='results', pattern='results*.json'):
        self.directory = directory
        self.pattern = pattern
        if not os.path.exists('details_data'):
            os.makedirs('details_data')


    def _find_files(self, directory='results', pattern='results*.json'):
        matching_files = []  # List to hold matching filenames
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    matching_files.append(filename)  # Append the matching filename to the list
        return matching_files  # Return the list of matching filenames

    @staticmethod
    def download_file(url, directory='details_data'):
        # Extract relevant parts from the URL
        segments = url.split('/')
        organization = segments[-3]
        model_name = segments[-2]
        task = url.split('%7ChendrycksTest-')[1].split('%7C')[0]

        # Construct the filename
        safe_file_name = f"{organization}_{model_name}_{task}.json"

        # Create the full save file path
        save_file_path = os.path.join(directory, safe_file_name)

        error_count = 0
        success_count = 0
        try:
            # Sending a GET request
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status()

            # Writing the content to the specified file
            with open(save_file_path, 'wb') as file:
                file.write(r.content)
            print(save_file_path)

            success_count += 1
        except requests.ConnectionError as e:
            error_count += 1
        except requests.HTTPError as e:
            error_count += 1
        except FileNotFoundError as e:
            error_count += 1
        except Exception as e:
            error_count += 1

        return error_count, success_count



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

        try:
            timestamp = segments[3].split('_')[1]
        except IndexError:
            print(f"Error: {file_path}")
            return None
        
        url = f'https://huggingface.co/datasets/open-llm-leaderboard/details/resolve/main/{bits}/{model_name}/details_harness%7ChendrycksTest-moral_scenarios%7C5_{quote(timestamp, safe="")}'
        return url
    
    def pipeline(self):
        error_count = 0
        success_count = 0
        file_paths = self._find_files(self.directory, self.pattern)
        for file_path in file_paths:
            print(f"Processing file path: {file_path}")
            url = self.build_url(file_path)
            if url:
                errors, successes = self.download_file(url)
                error_count += errors
                success_count += successes
            else:
                print(f"Error building URL for file path: {file_path}")
                error_count += 1

        print(f"Downloaded {success_count} files successfully. Encountered {error_count} errors.")
        return success_count, error_count

