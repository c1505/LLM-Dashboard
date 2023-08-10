import streamlit as st
import pandas as pd
import plotly.express as px
from result_data_processor import ResultDataProcessor
import matplotlib.pyplot as plt
import numpy as np


def plot_top_n(df, target_column, n=10):
    top_n = df.nlargest(n, target_column)

    # Initialize the bar plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Set width for each bar and their positions
    width = 0.28
    ind = np.arange(len(top_n))

    # Plot target_column and MMLU_average on the primary y-axis with adjusted positions
    ax1.bar(ind - width, top_n[target_column], width=width, color='blue', label=target_column)
    ax1.bar(ind, top_n['MMLU_average'], width=width, color='orange', label='MMLU_average')

    # Set the primary y-axis labels and title
    ax1.set_title(f'Top {n} performing models on {target_column}')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')

    # Create a secondary y-axis for Parameters
    ax2 = ax1.twinx()

    # Plot Parameters as bars on the secondary y-axis with adjusted position
    ax2.bar(ind + width, top_n['Parameters'], width=width, color='red', label='Parameters')

    # Set the secondary y-axis labels
    ax2.set_ylabel('Parameters', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Set the x-ticks and their labels
    ax1.set_xticks(ind)
    ax1.set_xticklabels(top_n.index, rotation=45, ha="right")

    # Adjust the legend
    fig.tight_layout()
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show the plot
    st.pyplot(fig)

data_provider = ResultDataProcessor()

# st.title('Model Evaluation Results including MMLU by task')
st.title('MMLU-by-Task Evaluation Results for 700+ Open Source Models')
st.markdown("""***Last updated August 7th***""")
st.markdown("""
            Hugging Face has run evaluations on over 500 open source models and provides results on a
            [publicly available leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) and [dataset](https://huggingface.co/datasets/open-llm-leaderboard/results). 
            The leaderboard currently displays the overall result for MMLU. This page shows individual accuracy scores for all 57 tasks of the MMLU evaluation.
            [Preliminary analysis of MMLU-by-Task data](https://coreymorrisdata.medium.com/preliminary-analysis-of-mmlu-evaluation-data-insights-from-500-open-source-models-e67885aa364b)
            """)

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

filtered_data.index.name = "Model Name"

csv = filtered_data.to_csv(index=True)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="model_evaluation_results.csv",
    mime="text/csv",
)

def create_plot(df, x_values, y_values, models=None, title=None):
    if models is not None:
        df = df[df.index.isin(models)]

    # remove rows with NaN values
    df = df.dropna(subset=[x_values, y_values])

    plot_data = pd.DataFrame({
        'Model': df.index,
        x_values: df[x_values],
        y_values: df[y_values],
    })

    plot_data['color'] = 'purple'
    fig = px.scatter(plot_data, x=x_values, y=y_values, color='color', hover_data=['Model'], trendline="ols")
    
    # If title is not provided, use x_values vs. y_values as the default title
    if title is None:
        title = x_values + " vs. " + y_values
    
    layout_args = dict(
        showlegend=False, 
        xaxis_title=x_values,
        yaxis_title=y_values,
        xaxis=dict(),
        yaxis=dict(),
        title=title
    )
    fig.update_layout(**layout_args)
    
    # Add a dashed line at 0.25 for the y_values
    x_min = df[x_values].min()
    x_max = df[x_values].max()

    y_min = df[y_values].min()
    y_max = df[y_values].max()

    if x_values.startswith('MMLU'): 
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

    if y_values.startswith('MMLU'):
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
st.write("As expected, there is a strong positive relationship between the number of parameters and average performance on the MMLU evaluation.")
st.markdown("***The dashed red line indicates random chance accuracy of 0.25 as the MMLU evaluation is multiple choice with 4 response options.***")
selected_x_column = st.selectbox('Select x-axis', filtered_data.columns.tolist(), index=0)
selected_y_column = st.selectbox('Select y-axis', filtered_data.columns.tolist(), index=3)

if selected_x_column != selected_y_column:    # Avoid creating a plot with the same column on both axes
    fig = create_plot(filtered_data, selected_x_column, selected_y_column)
    st.plotly_chart(fig)
else:
    st.write("Please select different columns for the x and y axes.")

# end of custom scatter plots
st.markdown("## Notable findings and plots")

st.markdown('### Abstract Algebra Performance')
st.write("Small models showed surprisingly strong performance on the abstract algebra task.  A 6 Billion parameter model is tied for the best performance on this task and there are a number of other small models in the top 10.")
plot_top_n(filtered_data, 'MMLU_abstract_algebra', 10)

fig = create_plot(filtered_data, 'Parameters', 'MMLU_abstract_algebra')
st.plotly_chart(fig)

# Moral scenarios plots
st.markdown("### Moral Scenarios Performance")
st.write("""
         While smaller models can perform well at many tasks, the model size threshold for decent performance on moral scenarios is much higher.  
         There are no models with less than 13 billion parameters with performance much better than random chance. Further investigation into other capabilities that emerge at 13 billion parameters could help
         identify capabilities that are important for moral reasoning.
            """)

fig = create_plot(filtered_data, 'Parameters', 'MMLU_moral_scenarios', title="Impact of Parameter Count on Accuracy for Moral Scenarios")
st.plotly_chart(fig)
st.write()

fig = create_plot(filtered_data, 'MMLU_average', 'MMLU_moral_scenarios')
st.plotly_chart(fig)

st.markdown("***Thank you to hugging face for running the evaluations and supplying the data as well as the original authors of the evaluations.***")

st.markdown("""
# References

1. Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, Thomas Wolf. (2023). *Open LLM Leaderboard*. Hugging Face. [link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

2. Gao, Leo et al. (2021). *A framework for few-shot language model evaluation*. Zenodo. [link](https://doi.org/10.5281/zenodo.5371628)

3. Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord. (2018). *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv. [link](https://arxiv.org/abs/1803.05457)

4. Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, Yejin Choi. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?*. arXiv. [link](https://arxiv.org/abs/1905.07830)

5. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt. (2021). *Measuring Massive Multitask Language Understanding*. arXiv. [link](https://arxiv.org/abs/2009.03300)

6. Stephanie Lin, Jacob Hilton, Owain Evans. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. arXiv. [link](https://arxiv.org/abs/2109.07958)
""")
