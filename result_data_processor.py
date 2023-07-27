import pandas as pd
import os
import fnmatch
import json
import re
import numpy as np

class ResultDataProcessor:
    
    def __init__(self, directory='results', pattern='results*.json'):
        self.directory = directory
        self.pattern = pattern
        self.data = self.process_data()

    @staticmethod
    def _find_files(directory, pattern):
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename
    
    def _read_and_transform_data(self, filename):
        with open(filename) as f:
            data = json.load(f)
        df = pd.DataFrame(data['results']).T
        return df
    
    def _cleanup_dataframe(self, df, model_name):
        df = df.rename(columns={'acc': model_name})
        df.index = (df.index.str.replace('hendrycksTest-', 'MMLU_', regex=True)
                          .str.replace('harness\|', '', regex=True)
                          .str.replace('\|5', '', regex=True))
        return df[[model_name]]
    
    @staticmethod
    def _extract_parameters(model_name):
        """
        Function to extract parameters from model name.
        It handles names with 'b/B' for billions and 'm/M' for millions. 
        """
        # pattern to match a number followed by 'b' (representing billions) or 'm' (representing millions)
        pattern = re.compile(r'(\d+\.?\d*)([bBmM])')
        
        match = pattern.search(model_name)
        
        if match:
            num, magnitude = match.groups()
            num = float(num)
            
            # convert millions to billions
            if magnitude.lower() == 'm':
                num /= 1000
            
            return num
        
        # return NaN if no match
        return np.nan

    
    def process_data(self):
        dataframes = [self._cleanup_dataframe(self._read_and_transform_data(filename), filename.split('/')[2])
                      for filename in self._find_files(self.directory, self.pattern)]

        data = pd.concat(dataframes, axis=1).transpose()
        
        # Add Model Name and rearrange columns
        data['Model Name'] = data.index
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        # Remove the 'Model Name' column
        data = data.drop(columns=['Model Name'])
        
        # Add average column
        data['MMLU_average'] = data.filter(regex='MMLU').mean(axis=1)

        # Reorder columns to move 'MMLU_average' to the third position
        cols = data.columns.tolist()
        cols = cols[:2] + cols[-1:] + cols[2:-1]
        data = data[cols]

        # Drop specific columns
        data.drop(columns=['all', 'truthfulqa:mc|0'])


        # Add parameter count column using extract_parameters function
        data['Parameters'] = data.index.to_series().apply(self._extract_parameters)

        # move the parameters column to the front of the dataframe
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        return data

    def get_data(self, selected_models):
        return self.data[self.data.index.isin(selected_models)]
