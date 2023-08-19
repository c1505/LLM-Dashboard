import pytest
import pandas as pd
import unittest
from result_data_processor import ResultDataProcessor
import os

class TestRegression(unittest.TestCase):
    def test_data_output_is_the_same(self):

        df_current = ResultDataProcessor().data

        # load the reference dataframe
        last_commit = os.popen('git rev-parse HEAD').read().strip()
        print(last_commit)
        reference_file = f'dataframe_history/output_{last_commit}.parquet'
        df_reference = pd.read_parquet(reference_file)


        # Compare DataFrames, allowing for some tolerance in floating-point comparisons
        pd.testing.assert_frame_equal(df_current, df_reference, check_dtype=True, atol=1e-5)


if __name__ == '__main__':
    unittest.main()