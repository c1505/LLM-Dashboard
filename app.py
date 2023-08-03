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
# name the index to include in the csv download


filtered_data.index.name = "Model Name"

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

    # remove rows with NaN values
    df = df.dropna(subset=[arc_column, moral_column])

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
    
    # Add a dashed line at 0.25 for the moral columns
    x_min = df[arc_column].min()
    x_max = df[arc_column].max()

    y_min = df[moral_column].min()
    y_max = df[moral_column].max()

    if arc_column.startswith('MMLU'): 
        fig.add_shape(
        type='line',
        x0=0.25, x1=0.25,
        y0=y_min, y1=y_max,
        line=dict(
            color='red',
            width=2,
            dash='dash'
        )
        )

    if moral_column.startswith('MMLU'):
        fig.add_shape(
        type='line',
        x0=x_min, x1=x_max,
        y0=0.25, y1=0.25,
        line=dict(
            color='red',
            width=2,
            dash='dash'
        )
        )

    
    return fig

# Custom scatter plots
st.header('Custom scatter plots')
selected_x_column = st.selectbox('Select x-axis', filtered_data.columns.tolist(), index=0)
selected_y_column = st.selectbox('Select y-axis', filtered_data.columns.tolist(), index=1)

if selected_x_column != selected_y_column:    # Avoid creating a plot with the same column on both axes
    fig = create_plot(filtered_data, selected_x_column, selected_y_column)
    st.plotly_chart(fig)
else:
    st.write("Please select different columns for the x and y axes.")

# end of custom scatter plots

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


st.markdown("**Thank you to hugging face for running the evaluations and supplying the data as well as the original authors of the evaluations**")


st.markdown("""
# References

1. Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, Thomas Wolf. (2023). *Open LLM Leaderboard*. Hugging Face. [link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

2. Gao, Leo et al. (2021). *A framework for few-shot language model evaluation*. Zenodo. [link](https://doi.org/10.5281/zenodo.5371628)

3. Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord. (2018). *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv. [link](https://arxiv.org/abs/1803.05457)

4. Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, Yejin Choi. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?*. arXiv. [link](https://arxiv.org/abs/1905.07830)

5. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt. (2021). *Measuring Massive Multitask Language Understanding*. arXiv. [link](https://arxiv.org/abs/2009.03300)

6. Stephanie Lin, Jacob Hilton, Owain Evans. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. arXiv. [link](https://arxiv.org/abs/2109.07958)
""")
