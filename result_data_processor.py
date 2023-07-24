import pandas as pd
import os
import fnmatch
import json

class ResultDataProcessor:
    def __init__(self):
        self.data = self.process_data()

    def process_data(self):
        dataframes = []

        def find_files(directory, pattern):
            for root, dirs, files in os.walk(directory):
                for basename in files:
                    if fnmatch.fnmatch(basename, pattern):
                        filename = os.path.join(root, basename)
                        yield filename

        for filename in find_files('results', 'results*.json'):
            model_name = filename.split('/')[2]
            with open(filename) as f:
                data = json.load(f)
                df = pd.DataFrame(data['results']).T


                # data cleanup
                df = df.rename(columns={'acc': model_name})
                # Replace 'hendrycksTest-' with a more descriptive column name
                df.index = df.index.str.replace('hendrycksTest-', 'MMLU_', regex=True)
                df.index = df.index.str.replace('harness\|', '', regex=True)
                # remove |5 from the index
                df.index = df.index.str.replace('\|5', '', regex=True)


                dataframes.append(df[[model_name]])

        data = pd.concat(dataframes, axis=1)

        data = data.transpose()
        data['Model Name'] = data.index
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        # remove the Model Name column
        data = data.drop(['Model Name'], axis=1)

        # remove the all column
        data = data.drop(['all'], axis=1)

        # remove the truthfulqa:mc|0 column
        data = data.drop(['truthfulqa:mc|0'], axis=1)

        # create a new column that averages the results from each of the columns with a name that start with MMLU
        data['MMLU_average'] = data.filter(regex='MMLU').mean(axis=1)

        # move the MMLU_average column to the third column in the dataframe
        cols = data.columns.tolist()
        cols = cols[:2] + cols[-1:] + cols[2:-1]
        data = data[cols]

        return data
    
    # filter data based on the index
    def get_data(self, selected_models):
        filtered_data = self.data[self.data.index.isin(selected_models)]
        return filtered_data