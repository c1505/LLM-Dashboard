import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

def load_csv_data(file_path):
    return pd.read_csv(file_path)





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

# Function to create an unfilled radar chart
def create_radar_chart_unfilled(df, model_names, metrics):
    fig = go.Figure()
    min_value = df.loc[model_names, metrics].min().min()
    max_value = df.loc[model_names, metrics].max().max()
    for model_name in model_names:
        values_model = df.loc[model_name, metrics]
        fig.add_trace(go.Scatterpolar(
            r=values_model,
            theta=metrics,
            name=model_name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min_value, max_value]
            )),
        showlegend=True,
        width=800,  # Change the width as needed
        height=600   # Change the height as needed
    )
    return fig



# Function to create a line chart
def create_line_chart(df, model_names, metrics):
    line_data = []
    for model_name in model_names:
        values_model = df.loc[model_name, metrics]
        for metric, value in zip(metrics, values_model):
            line_data.append({'Model': model_name, 'Metric': metric, 'Value': value})

    line_df = pd.DataFrame(line_data)

    fig = px.line(line_df, x='Metric', y='Value', color='Model', title='Comparison of Models', line_dash_sequence=['solid'])
    fig.update_layout(showlegend=True)
    return fig

def find_top_differences_table(df, target_model, closest_models, num_differences=10, exclude_columns=['Parameters']):
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

# st.title('Model Evaluation Results including MMLU by task')
st.title('Interactive Portal for Analyzing Open Source Large Language Models')
st.markdown("""***Last updated March 17th 2024***""")
st.markdown("""**It has not been updated to correctly extract the parameter number from mixture of experts models.**""")
st.markdown("""**As of 04-17-2024, this data was not generated using the chat templates. Smaller models are especially sensative to this and other aspects related to the format of the inputs.**""")
st.markdown("""
            This page provides a way to explore the results for individual tasks and compare models across tasks. Data for the benchmarks hellaswag, arc_challenge, and truthfulQA have also been included for comparison. 
            There are 57 tasks in the MMLU evaluation that cover a wide variety of subjects including Science, Math, Humanities, Social Science, Applied Science, Logic, and Security.
            [Preliminary analysis of MMLU-by-Task data](https://coreymorrisdata.medium.com/preliminary-analysis-of-mmlu-evaluation-data-insights-from-500-open-source-models-e67885aa364b)
            """)

# Load the data into memory
data_path = "processed_data_2024-04-16.csv"
data_df = load_csv_data(data_path)
# drop the column Unnamed: 0
data_df.rename(columns={'Unnamed: 0': "Model Name"}, inplace=True)
data_df.set_index("Model Name", inplace=True)

filtered_data = data_df

# sort the table by the MMLU_average column
filtered_data = filtered_data.sort_values(by=['MMLU_average'], ascending=False)

# Select box for filtering by Parameters
parameter_threshold = st.selectbox(
    'Filter by Parameters (Less Than or Equal To):',
    options=[3, 7, 13, 35, 'No threshold'],
    index=4,  # Set the default selected option to 'No threshold'
    format_func=lambda x: f"{x}" if isinstance(x, int) else x
)
if isinstance(parameter_threshold, int):
    filtered_data = filtered_data[filtered_data['Parameters'] <= parameter_threshold]

# model name filtering
search_queries = st.text_input("Filter by Model Name:", "").replace(" ", "").split(',')
if search_queries:
    filtered_data = filtered_data[filtered_data.index.str.contains('|'.join(search_queries), case=False)]

# column name filtering
column_search_query = st.text_input("Filter by Column/Task Name:", "").replace(" ", "").split(',')
matching_columns = [col for col in filtered_data.columns if any(query.lower() in col.lower() for query in column_search_query)]
filtered_data = filtered_data[matching_columns]


# Display the DataFrame with only the matching columns
st.markdown("## Sortable Results")
st.dataframe(
    filtered_data[matching_columns],
    column_config={
        "URL": st.column_config.LinkColumn( # Only current way to make url a clickable link with streamlit without removing the interactivity of the table
            width="small"
        )
    },
    hide_index=True,
)

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

    #remove label rows URL, full_model_name
    df = df.drop(columns=['URL', 'full_model_name'])

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
        title=title,
        height=500,
        width=1000,
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
st.write("""
         The scatter plot is useful to identify models that outperform or underperform on a particular task in relation to their size or overall performance.
         Identifying these models is a first step to better understand what training strategies result in better performance on a particular task.
         """)
st.markdown("***The dashed red line indicates random chance accuracy of 0.25 as the MMLU evaluation is multiple choice with 4 response options.***")
# add a line separating the writing
st.markdown("***")
st.write("As expected, there is a strong positive relationship between the number of parameters and average performance on the MMLU evaluation.")


column_list_for_plotting = filtered_data.columns.tolist()
if 'URL' in column_list_for_plotting:
    column_list_for_plotting.remove('URL')
if 'full_model_name' in column_list_for_plotting:
    column_list_for_plotting.remove('full_model_name')

selected_x_column = st.selectbox('Select x-axis', column_list_for_plotting, index=0)
selected_y_column = st.selectbox('Select y-axis', column_list_for_plotting, index=1)

if selected_x_column != selected_y_column:    # Avoid creating a plot with the same column on both axes
    fig = create_plot(filtered_data, selected_x_column, selected_y_column)
    st.plotly_chart(fig)
else:
    st.write("Please select different columns for the x and y axes.")


# end of custom scatter plots



# # Section to select a model and display radar and line charts
# st.header("Compare a Selected Model to the 5 Models Closest in MMLU Average Performance")
# st.write("""
#          This comparison highlights the nuances in model performance across different tasks. 
#          While the overall MMLU average score provides a general understanding of a model's capabilities, 
#          examining the closest models reveals variations in performance on individual tasks. 
#          Such an analysis can uncover specific strengths and weaknesses and guide further exploration and improvement.
#          """)

# default_model_name = "GPT-JT-6B-v0"

# default_model_index = filtered_data.index.tolist().index(default_model_name) if default_model_name in filtered_data.index else 0
# selected_model_name = st.selectbox("Select a Model:", filtered_data.index.tolist(), index=default_model_index)

# # Get the closest 5 models with unique indices
# closest_models_diffs = filtered_data['MMLU_average'].sub(filtered_data.loc[selected_model_name, 'MMLU_average']).abs()
# closest_models = closest_models_diffs.nsmallest(5, keep='first').index.drop_duplicates().tolist()


# Find the top 10 tasks with the largest differences and convert to a DataFrame
# top_differences_table, top_differences_tasks = find_top_differences_table(filtered_data, selected_model_name, closest_models)

# Display the DataFrame for the closest models and the top differences tasks
# st.dataframe(filtered_data.loc[closest_models, top_differences_tasks])

# # Display the table in the Streamlit app
# st.markdown("## Top Differences")
# st.dataframe(top_differences_table)

# Create a radar chart for the tasks with the largest differences
# fig_radar_top_differences = create_radar_chart_unfilled(filtered_data, closest_models, top_differences_tasks)

# Display the radar chart
# st.plotly_chart(fig_radar_top_differences)


st.markdown("## Notable findings and plots")

# Moral scenarios plots
st.markdown("### MMLU’s Moral Scenarios Benchmark Doesn’t Measure What You Think it Measures")
def show_random_moral_scenarios_question():
    moral_scenarios_data = pd.read_csv('moral_scenarios_questions.csv')
    random_question = moral_scenarios_data.sample()
    expander = st.expander("Show a random moral scenarios question")
    expander.write(random_question['query'].values[0])



st.write("""
         After a deeper dive into the moral scenarios task, it appears that benchmark is not a valid measurement of moral judgement.
         The challenges these models face are not rooted in understanding each scenario, but rather in the structure of the task itself.
         I would recommend using a different benchmark for moral judgement. More details of the analysis can be found here: [MMLU’s Moral Scenarios Benchmark Doesn’t Measure What You Think it Measures ](https://medium.com/p/74fd6e512521)
            """)

show_random_moral_scenarios_question()

fig = create_plot(filtered_data, 'Parameters', 'MMLU_moral_scenarios', title="Impact of Parameter Count on Accuracy for Moral Scenarios")
st.plotly_chart(fig)
st.write()



fig = create_plot(filtered_data, 'MMLU_average', 'MMLU_moral_scenarios')
st.plotly_chart(fig)

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
