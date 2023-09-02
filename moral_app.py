import streamlit as st
import pandas as pd
import plotly.express as px
from result_data_processor import ResultDataProcessor
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotting_utils import plot_top_n, create_radar_chart_unfilled, create_line_chart, create_plot

st.set_page_config(layout="wide")

def find_top_differences_table(df, target_model, closest_models, num_differences=10, exclude_columns=['Parameters', 'organization']):
    # Calculate the absolute differences for each task between the target model and the closest models
    new_df = df.drop(columns=exclude_columns)
    differences = new_df.loc[closest_models].sub(new_df.loc[target_model]).abs()
    # Unstack the differences and sort by the largest absolute difference
    top_differences = differences.unstack().nlargest(num_differences)
    # Convert the top differences to a DataFrame for display
    top_differences_table = pd.DataFrame({
        'Task': [idx[0] for idx in top_differences.index],
        'Difference': top_differences.values
    })
    # Ensure that only unique tasks are returned
    unique_top_differences_tasks = list(set(top_differences_table['Task'].tolist()))
    return top_differences_table, unique_top_differences_tasks



# Main Application

data_provider = ResultDataProcessor()

st.title('Why are large language models so bad at the moral scenarios task?')
st.markdown("""
            Here I am to answer the question: Why are large language models so bad at the moral scenarios task?
            Sub questions: 
            - Are the models actually bad at moral reasoning ?
            - Is it the structure of the task that is the causing the poor performance ?
                - Are there other tasks with questions in a similar structure ? 
                - How do models perform when the structure of the task is changed ?  
            """)

filters = st.checkbox('Select Models and/or Evaluations')

# Initialize selected columns with "Parameters" and "MMLU_average" if filters are checked
selected_columns = ['Parameters', 'MMLU_average'] if filters else data_provider.data.columns.tolist()

# Initialize selected models as empty if filters are checked
selected_models = [] if filters else data_provider.data.index.tolist()

if filters:
    # Create multi-select for columns with default selection
    selected_columns = st.multiselect(
        'Select Columns',
        data_provider.data.columns.tolist(),
        default=selected_columns
    )

    # Create multi-select for models without default selection
    selected_models = st.multiselect(
        'Select Models',
        data_provider.data.index.tolist()
    )

# Get the filtered data
filtered_data = data_provider.get_data(selected_models)

# sort the table by the MMLU_average column
filtered_data = filtered_data.sort_values(by=['MMLU_average'], ascending=False)

# Select box for filtering by Parameters
parameter_threshold = st.selectbox(
    'Filter by Parameters (Less Than or Equal To):',
    options=[3, 7, 13, 35, 'No threshold'],
    index=4,  # Set the default selected option to 'No threshold'
    format_func=lambda x: f"{x}" if isinstance(x, int) else x
)

# Filter the DataFrame based on the selected parameter threshold if not 'No threshold'
if isinstance(parameter_threshold, int):
    filtered_data = filtered_data[filtered_data['Parameters'] <= parameter_threshold]


# Search box
search_query = st.text_input("Filter by Model Name:", "")

# Filter the DataFrame based on the search query in the index (model name)
if search_query:
    filtered_data = filtered_data[filtered_data.index.str.contains(search_query, case=False)]


# Search box for columns
column_search_query = st.text_input("Filter by Column/Task Name:", "")

# Get the columns that contain the search query
matching_columns = [col for col in filtered_data.columns if column_search_query.lower() in col.lower()]

# # Display the DataFrame with only the matching columns
# st.markdown("## Sortable Results")
# st.dataframe(filtered_data[matching_columns])


# CSV download

filtered_data.index.name = "Model Name"

csv = filtered_data.to_csv(index=True)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="model_evaluation_results.csv",
    mime="text/csv",
)


# Moral Scenarios section
st.markdown("## Why are large language models so bad at the moral scenarios task?")
st.markdown("### The structure of the task is odd")

# - Are the models actually bad at moral reasoning ?
# - Is it the structure of the task that is the causing the poor performance ?
#     - Are there other tasks with questions in a similar structure ? 
#     - How do models perform when the structure of the task is changed ?  
st.markdown("### Moral Scenarios Performance")
def show_random_moral_scenarios_question():
    moral_scenarios_data = pd.read_csv('moral_scenarios_questions.csv')
    random_question = moral_scenarios_data.sample()
    expander = st.expander("Show a random moral scenarios question")
    expander.write(random_question['query'].values[0])

show_random_moral_scenarios_question()

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








# Custom scatter plots
st.header('Custom scatter plots')
st.write("""
         The scatter plot is useful to identify models that outperform or underperform on a particular task in relation to their size or overall performance.
         Identifying these models is a first step to better understand what training strategies result in better performance on a particular task.
         """)
st.markdown("***The dashed red line indicates random chance accuracy of 0.25 as the MMLU evaluation is multiple choice with 4 response options.***")
# add a line separating the writing
st.markdown("***")
st.write("As expected, there is a strong positive relationship between the number of parameters and average performance on the MMLU evaluation.")

selected_x_column = st.selectbox('Select x-axis', filtered_data.columns.tolist(), index=0)
selected_y_column = st.selectbox('Select y-axis', filtered_data.columns.tolist(), index=3)

if selected_x_column != selected_y_column:    # Avoid creating a plot with the same column on both axes
    fig = create_plot(filtered_data, selected_x_column, selected_y_column)
    st.plotly_chart(fig)
else:
    st.write("Please select different columns for the x and y axes.")




# end of custom scatter plots

# Section to select a model and display radar and line charts
st.header("Compare a Selected Model to the 5 Models Closest in MMLU Average Performance")
st.write("""
         This comparison highlights the nuances in model performance across different tasks. 
         While the overall MMLU average score provides a general understanding of a model's capabilities, 
         examining the closest models reveals variations in performance on individual tasks. 
         Such an analysis can uncover specific strengths and weaknesses and guide further exploration and improvement.
         """)

default_model_name = "GPT-JT-6B-v0"

default_model_index = filtered_data.index.tolist().index(default_model_name) if default_model_name in filtered_data.index else 0
selected_model_name = st.selectbox("Select a Model:", filtered_data.index.tolist(), index=default_model_index)

# Get the closest 5 models with unique indices
closest_models_diffs = filtered_data['MMLU_average'].sub(filtered_data.loc[selected_model_name, 'MMLU_average']).abs()
closest_models = closest_models_diffs.nsmallest(5, keep='first').index.drop_duplicates().tolist()


# Find the top 10 tasks with the largest differences and convert to a DataFrame
top_differences_table, top_differences_tasks = find_top_differences_table(filtered_data, selected_model_name, closest_models)

# Display the DataFrame for the closest models and the top differences tasks
st.dataframe(filtered_data.loc[closest_models, top_differences_tasks])

# # Display the table in the Streamlit app
# st.markdown("## Top Differences")
# st.dataframe(top_differences_table)

# Create a radar chart for the tasks with the largest differences
fig_radar_top_differences = create_radar_chart_unfilled(filtered_data, closest_models, top_differences_tasks)

# Display the radar chart
st.plotly_chart(fig_radar_top_differences)


st.markdown("## Notable findings and plots")

st.markdown('### Abstract Algebra Performance')
st.write("Small models showed surprisingly strong performance on the abstract algebra task.  A 6 Billion parameter model is tied for the best performance on this task and there are a number of other small models in the top 10.")
plot_top_n(filtered_data, 'MMLU_abstract_algebra', 10)

fig = create_plot(filtered_data, 'Parameters', 'MMLU_abstract_algebra')
st.plotly_chart(fig)






st.markdown("***Thank you to hugging face for running the evaluations and supplying the data as well as the original authors of the evaluations.***")

st.markdown("""
# Citation

1. Corey Morris (2023). *Exploring the Characteristics of Large Language Models: An Interactive Portal for Analyzing 700+ Open Source Models Across 57 Diverse Evaluation Tasks*. [link](https://huggingface.co/spaces/CoreyMorris/MMLU-by-task-Leaderboard)
            
2. Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, Thomas Wolf. (2023). *Open LLM Leaderboard*. Hugging Face. [link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

3. Gao, Leo et al. (2021). *A framework for few-shot language model evaluation*. Zenodo. [link](https://doi.org/10.5281/zenodo.5371628)

4. Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord. (2018). *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*. arXiv. [link](https://arxiv.org/abs/1803.05457)

5. Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, Yejin Choi. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?*. arXiv. [link](https://arxiv.org/abs/1905.07830)

6. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt. (2021). *Measuring Massive Multitask Language Understanding*. arXiv. [link](https://arxiv.org/abs/2009.03300)

7. Stephanie Lin, Jacob Hilton, Owain Evans. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. arXiv. [link](https://arxiv.org/abs/2109.07958)
""")
