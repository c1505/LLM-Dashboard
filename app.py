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

    def get_top_performing_models(self, column_name, top_n):
        sorted_data = self.data.sort_values(by=column_name, ascending=False)
        return sorted_data['Model Name'].head(top_n).tolist()

    def get_data(self, selected_models):
        filtered_data = self.data[self.data['Model Name'].isin(selected_models)]
        return filtered_data
    

data_provider = MultiURLData()

# Get top 20 performing models
top_models = data_provider.get_top_performing_models('harness|arc:challenge|25', 20)

# Initialize selected models and columns
selected_models = top_models
selected_columns = data_provider.data.columns.tolist()

# Create placeholders for the dataframe and multiselects
df_placeholder = st.empty()
models_multiselect = st.empty()
columns_multiselect = st.empty()

# Function to display dataframe
def display_dataframe(models, columns):
    filtered_data = data_provider.get_data(models)
    filtered_data = filtered_data[columns]
    df_placeholder.dataframe(filtered_data)

# Function to display multiselects
def display_multiselects():
    models = models_multiselect.multiselect(
        'Select Models',
        data_provider.data['Model Name'].tolist(),
        default=selected_models
    )
    columns = columns_multiselect.multiselect(
        'Select Columns',
        data_provider.data.columns.tolist(),
        default=selected_columns
    )
    return models, columns

# Display dataframe initially
display_dataframe(selected_models, selected_columns)

# Display multiselects initially
selected_models, selected_columns = display_multiselects()

# If the user clicks the "Update" button, update the selected models and columns
# and redisplay the dataframe
if st.button('Update'):
    display_dataframe(selected_models, selected_columns)
