import pytest
import pandas as pd
import unittest
from result_data_processor import ResultDataProcessor
import os

class TestRegression(unittest.TestCase):
    def test_data_output_is_the_same(self):

        # # Run code and get output
        # df = ResultDataProcessor().data

        # # Save output to CSV
        # commit = '5603e9f51b643dbbb3da62943d5f26c48e11ae9b'
        # outfile = f'output_{commit}.csv'
        # df.to_csv(outfile)

        # # Checkout previous commit
        # prev_commit = '02b17021c8c2759ae08d23578b11bdb493941479'
        # checkout_command = f'git checkout {prev_commit}' 
        # os.system(checkout_command)

        # # Rerun and save output
        # df = ResultDataProcessor().data
        # prev_outfile = f'output_{prev_commit}.csv'
        # prev_df.to_csv(prev_outfile)

        # # Diff outputs
        # import filecmp
        # assert filecmp.cmp(outfile, prev_outfile)

        df_current = ResultDataProcessor().data

        # Load the reference output file (generated manually from a previous commit)
        commit = '5b83d0bbaf92089fd7713e9e8edaa14c821b0dc7'
        reference_file = f'output_{commit}.csv'
        df_reference = pd.read_csv(reference_file)

        # Compare DataFrames, allowing for some tolerance in floating-point comparisons
        pd.testing.assert_frame_equal(df_current, df_reference, check_dtype=False, atol=1e-5)


if __name__ == '__main__':
    unittest.main()