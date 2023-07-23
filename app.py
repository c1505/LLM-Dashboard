import streamlit as st
import pandas as pd
import os
import fnmatch
import json
import plotly.express as px

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

                df.index = df.index.str.replace('hendrycksTest-', '', regex=True)

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

def create_plot(df, model_column, arc_column, moral_column, models=None):
    # Filter the dataframe if specific models are provided
    if models is not None:
        df = df[df[model_column].isin(models)]

    # Create a plot with new data
    plot_data = pd.DataFrame({
        'Model': list(df[model_column]),
        arc_column: list(df[arc_column]),
        moral_column: list(df[moral_column]),
    })

    # Calculate color column
    plot_data['color'] = 'purple'

    # # TODO maybe change this
    # plot_data.loc[plot_data[moral_column] < plot_data[arc_column], 'color'] = 'red'
    # plot_data.loc[plot_data[moral_column] > plot_data[arc_column], 'color'] = 'blue'

    # Create the scatter plot with trendline
    fig = px.scatter(plot_data, x=arc_column, y=moral_column, color='color', hover_data=['Model'], trendline="ols") #other option ols
    fig.update_layout(showlegend=False,  # hide legend
                    xaxis_title=arc_column,
                    yaxis_title=moral_column,
                    xaxis = dict(),
                    yaxis = dict())
    
    return fig


# models_to_plot = ['Model1', 'Model2', 'Model3']
# fig = create_plot(filtered_data, 'Model Name', 'arc:challenge|25', 'moral_scenarios|5', models=models_to_plot)

fig = create_plot(filtered_data, 'Model Name', 'arc:challenge|25', 'moral_scenarios')
st.plotly_chart(fig)

fig = create_plot(filtered_data, 'Model Name', 'arc:challenge|25', 'hellaswag|10')
st.plotly_chart(fig)

fig = create_plot(filtered_data, 'Model Name', 'moral_disputes', 'moral_scenarios')
st.plotly_chart(fig)
