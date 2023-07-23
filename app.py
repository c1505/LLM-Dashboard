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

st.title('Leaderboard')

# TODO actually use these checkboxes as filters
## Desired behavior
## model and column selection is hidden by default
## when the user clicks the checkbox, the model and column selection appears
filters = st.checkbox('Add filters')

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



#TODO fix this plot.  currently has an error
# Create a plot with new data
df = pd.DataFrame({
    'Model': list(filtered_data['Model Name']),
    'harness|arc:challenge|25_rank': list(filtered_data['harness|arc:challenge|25_rank']),
    'harness|moral_scenarios|5_rank': list(filtered_data['harness|moral_scenarios|5_rank']),
})

# Calculate color column
df['color'] = 'purple'
df.loc[df['harness|moral_scenarios|5_rank'] < df['harness|arc:challenge|25_rank'], 'color'] = 'red'
df.loc[df['harness|moral_scenarios|5_rank'] > df['harness|arc:challenge|25_rank'], 'color'] = 'blue'

# Create the scatter plot
fig = px.scatter(df, x='harness|arc:challenge|25_rank', y='harness|moral_scenarios|5_rank', color='color', hover_data=['Model'])
fig.update_layout(showlegend=False,  # hide legend
                  xaxis = dict(autorange="reversed"),  # reverse X-axis
                  yaxis = dict(autorange="reversed"))  # reverse Y-axis

# Show the plot in Streamlit
st.plotly_chart(fig)



