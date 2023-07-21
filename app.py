import streamlit as st
import pandas as pd
import os
import fnmatch
import json

class MultiURLData:
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

                df = df.rename(columns={'acc': model_name})

                df.index = df.index.str.replace('hendrycksTest-', '')

                df.index = df.index.str.replace('harness\\|', '')

                dataframes.append(df[[model_name]])

        data = pd.concat(dataframes, axis=1)

        data = data.transpose()
        data['Model Name'] = data.index
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        return data

    def get_data(self, selected_models):
        filtered_data = self.data[self.data['Model Name'].isin(selected_models)]
        return filtered_data

data_provider = MultiURLData()

# Create checkboxes for each column
selected_columns = st.multiselect(
    'Select Columns',
    data_provider.data.columns.tolist(),
    default=data_provider.data.columns.tolist()
)

selected_models = st.multiselect(
    'Select Models',
    data_provider.data['Model Name'].tolist(),
    default=data_provider.data['Model Name'].tolist()
)


# Get the filtered data and display it in a table
filtered_data = data_provider.get_data(selected_models)
st.dataframe(filtered_data)
