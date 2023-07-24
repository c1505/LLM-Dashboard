import streamlit as st
import pandas as pd
import os
import fnmatch
import json
import plotly.express as px

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

data_provider = ResultDataProcessor()

st.title('Model Evaluation Results including MMLU by task')

filters = st.checkbox('Select Models and Evaluations')

# Create defaults for selected columns and models
selected_columns = data_provider.data.columns.tolist()
selected_models = data_provider.data.index.tolist()

if filters:
    # Create checkboxes for each column
    selected_columns = st.multiselect(
        'Select Columns',
        data_provider.data.columns.tolist(),
        default=selected_columns
    )

    selected_models = st.multiselect(
        'Select Models',
        data_provider.data.index.tolist(),
        default=selected_models
    )

# Get the filtered data
st.header('Sortable table')
filtered_data = data_provider.get_data(selected_models)

# sort the table by the MMLU_average column
filtered_data = filtered_data.sort_values(by=['MMLU_average'], ascending=False)
st.dataframe(filtered_data[selected_columns])

# CSV download
csv = filtered_data.to_csv(index=True)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="model_evaluation_results.csv",
    mime="text/csv",
)


def create_plot(df, arc_column, moral_column, models=None):
    if models is not None:
        df = df[df.index.isin(models)]

    plot_data = pd.DataFrame({
        'Model': df.index,
        arc_column: df[arc_column],
        moral_column: df[moral_column],
    })

    plot_data['color'] = 'purple'
    fig = px.scatter(plot_data, x=arc_column, y=moral_column, color='color', hover_data=['Model'], trendline="ols")
    fig.update_layout(showlegend=False, 
                      xaxis_title=arc_column,
                      yaxis_title=moral_column,
                      xaxis = dict(),
                      yaxis = dict())
    
    return fig



st.header('Overall benchmark comparison')

st.header('Custom scatter plots')
selected_x_column = st.selectbox('Select x-axis', filtered_data.columns.tolist(), index=0)
selected_y_column = st.selectbox('Select y-axis', filtered_data.columns.tolist(), index=1)

if selected_x_column != selected_y_column:    # Avoid creating a plot with the same column on both axes
    fig = create_plot(filtered_data, selected_x_column, selected_y_column)
    st.plotly_chart(fig)
else:
    st.write("Please select different columns for the x and y axes.")

fig = create_plot(filtered_data, 'arc:challenge|25', 'hellaswag|10')
st.plotly_chart(fig)

fig = create_plot(filtered_data, 'arc:challenge|25', 'MMLU_average')
st.plotly_chart(fig)

fig = create_plot(filtered_data, 'hellaswag|10', 'MMLU_average')
st.plotly_chart(fig)

st.header('Top 50 models on MMLU_average')
top_50 = filtered_data.nlargest(50, 'MMLU_average')
fig = create_plot(top_50, 'arc:challenge|25', 'MMLU_average')
st.plotly_chart(fig)

st.header('Moral Scenarios')

fig = create_plot(filtered_data, 'arc:challenge|25', 'MMLU_moral_scenarios')
st.plotly_chart(fig)

fig = create_plot(filtered_data, 'MMLU_moral_disputes', 'MMLU_moral_scenarios')
st.plotly_chart(fig)

fig = create_plot(filtered_data, 'MMLU_average', 'MMLU_moral_scenarios')
st.plotly_chart(fig)

fig = px.histogram(filtered_data, x="MMLU_moral_scenarios", marginal="rug", hover_data=filtered_data.columns)
st.plotly_chart(fig)

fig = px.histogram(filtered_data, x="MMLU_moral_disputes", marginal="rug", hover_data=filtered_data.columns)
st.plotly_chart(fig)
