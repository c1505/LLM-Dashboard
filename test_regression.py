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

        #TODO
        # if there are no untracked changes, the dataframes should be the same
        # if there is no file saved for the current commit, save a file for the current commit
        # instead check the last commit to the one previous to that one
        # if there are untracked changes, the dataframes should be different
        # either optionally take a parameter for this test or extract the comparison logic so that it can be used separately to 
        # compare given any two commit hashes

        # Compare DataFrames, allowing for some tolerance in floating-point comparisons
        pd.testing.assert_frame_equal(df_current, df_reference, check_dtype=True, atol=1e-5)


if __name__ == '__main__':
    unittest.main()