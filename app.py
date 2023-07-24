import streamlit as st
import pandas as pd
import plotly.express as px
from result_data_processor import ResultDataProcessor

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

st.header('Custom scatter plots')
selected_x_column = st.selectbox('Select x-axis', filtered_data.columns.tolist(), index=0)
selected_y_column = st.selectbox('Select y-axis', filtered_data.columns.tolist(), index=1)

if selected_x_column != selected_y_column:    # Avoid creating a plot with the same column on both axes
    fig = create_plot(filtered_data, selected_x_column, selected_y_column)
    st.plotly_chart(fig)
else:
    st.write("Please select different columns for the x and y axes.")

st.header('Overall evaluation comparisons')

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

st.header('Moral Reasoning')

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
