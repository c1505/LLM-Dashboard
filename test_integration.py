import subprocess
import time
import requests
import unittest
from app import find_top_differences_table

class TestAppFunctions(unittest.TestCase):

    def setUp(self):
        # Assuming that you have a ResultDataProcessor class or equivalent that provides the data
        self.processor = ResultDataProcessor()  
        self.data = self.processor.data  # Assuming this gives you the DataFrame you need
    
    # def test_find_top_differences_table_error(self):
    #     # Define the variables that replicate the error
    #     closest_models = # define this based on your code
    #     exclude_columns = # define this based on your code
    #     selected_model_name = # define this based on your code
        
    #     # Run the problematic function without catching the TypeError
    #     top_differences_table, top_differences_tasks = find_top_differences_table(
    #         self.data, selected_model_name, closest_models, exclude_columns
    #     )
        
        # If you wish to add any assertions related to the expected output, add them here
    def test_streamlit_app_runs():
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
