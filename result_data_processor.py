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
        self.ranked_data = self.rank_data()

    def _find_files(self, directory='results', pattern='results*.json'):
        matching_files = {}
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    matching_files[root] = filename
        matching_files = {key: value for key, value in matching_files.items() if 'gpt-j-6b' not in key}
        matching_files = list(matching_files.values())
        return matching_files

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
    
    def _extract_mc1(self, df, model_name):
        df = df.rename(columns={'mc1': model_name})
        # rename row harness|truthfulqa:mc|0 to truthfulqa:mc1
        df.index = (df.index.str.replace('mc\|0', 'mc1', regex=True))
        # just return the harness|truthfulqa:mc1 row
        df = df.loc[['harness|truthfulqa:mc1']]
        return df[[model_name]]
    
    def _extract_mc2(self, df, model_name):
        # rename row harness|truthfulqa:mc|0 to truthfulqa:mc2
        df = df.rename(columns={'mc2': model_name})
        df.index = (df.index.str.replace('mc\|0', 'mc2', regex=True))
        df = df.loc[['harness|truthfulqa:mc2']]
        return df[[model_name]]
    
    # remove extreme outliers from column harness|truthfulqa:mc1
    def _remove_mc1_outliers(self, df):
        mc1 = df['harness|truthfulqa:mc1']
        # Identify the outliers
        # outliers_condition = mc1 > mc1.quantile(.95)
        outliers_condition = mc1 == 1.0
        # Replace the outliers with NaN
        df.loc[outliers_condition, 'harness|truthfulqa:mc1'] = np.nan
        return df


    
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
        
        dataframes = []
        organization_names = []
        for filename in self._find_files(self.directory, self.pattern):
            raw_data = self._read_and_transform_data(filename)
            split_path = filename.split('/')
            model_name = split_path[2]
            organization_name = split_path[1]
            cleaned_data = self._cleanup_dataframe(raw_data, model_name)
            mc1 = self._extract_mc1(raw_data, model_name)
            mc2 = self._extract_mc2(raw_data, model_name)
            cleaned_data = pd.concat([cleaned_data, mc1])
            cleaned_data = pd.concat([cleaned_data, mc2])
            organization_names.append(organization_name)
            dataframes.append(cleaned_data)


        data = pd.concat(dataframes, axis=1).transpose()

        # Add organization column
        data['organization'] = organization_names

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
        data = data.drop(columns=['all', 'truthfulqa:mc|0'])

        # Add parameter count column using extract_parameters function
        data['Parameters'] = data.index.to_series().apply(self._extract_parameters)

        # move the parameters column to the front of the dataframe
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        # remove extreme outliers from column harness|truthfulqa:mc1
        data = self._remove_mc1_outliers(data)

        data = self.manual_removal_of_models(data)

        return data
    
    def manual_removal_of_models(self, df):
    # remove models verified to be trained on evaluation data
        # load the list of models
        with open('contaminated_models.txt') as f:
            contaminated_models = f.read().splitlines()
        # remove the models from the dataframe
        df = df[~df.index.isin(contaminated_models)]
        return df

    
    def rank_data(self):
        # add rank for each column to the dataframe
        # copy the data dataframe to avoid modifying the original dataframe
        rank_data = self.data.copy()
        for col in list(rank_data.columns):
            rank_data[col + "_rank"] = rank_data[col].rank(ascending=False, method='min')

        return rank_data

    def get_data(self, selected_models):
        return self.data[self.data.index.isin(selected_models)]
