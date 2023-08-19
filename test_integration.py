import subprocess
import time
import requests
import unittest
from app import find_top_differences_table
from result_data_processor import ResultDataProcessor

class TestAppFunctions(unittest.TestCase):

    def setUp(self):
        # Assuming that you have a ResultDataProcessor class or equivalent that provides the data
        self.processor = ResultDataProcessor()  
        self.data = self.processor.data  # Assuming this gives you the DataFrame you need
    
    def test_find_top_differences_table_error(self):
        # replicating the error before fixing it
        filtered_data = self.data

        # Get the closest 5 models with unique indices
        selected_model_name = 'Platypus2-70B-instruct'
        exclude_columns=['Parameters','organization']
        closest_models_diffs = filtered_data['MMLU_average'].sub(filtered_data.loc[selected_model_name, 'MMLU_average']).abs()
        closest_models = closest_models_diffs.nsmallest(5, keep='first').index.drop_duplicates().tolist()




        # Run the problematic function without catching the TypeError
        top_differences_table, top_differences_tasks = find_top_differences_table(
            self.data, selected_model_name, closest_models, exclude_columns
        )
        
    def test_streamlit_app_runs(self):
        # Start the Streamlit app in a subprocess
        process = subprocess.Popen(["streamlit", "run", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for a few seconds to give Streamlit time to start
        time.sleep(5)

        # Make a request to the Streamlit app's default URL to check that it's running
        response = requests.get('http://localhost:8501')

        # Terminate the process
        process.terminate()

        # Check that the response from the Streamlit app was successful
        assert response.status_code == 200, "Streamlit app did not start successfully"

if __name__ == '__main__':
    unittest.main()
