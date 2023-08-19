# when run
# checks if there is uncommitted code
# if there is uncommitted code, ti retuns an error
# if there is no uncommitted code, it saves the dataframe as a parquet file with the commit hash in the name

import pytest
import pandas as pd
from result_data_processor import ResultDataProcessor

import os

import subprocess

def check_git_changes(repo_path):
    try:
        # Change to the repository directory
        original_path = os.getcwd()
        os.chdir(repo_path)

        # Run the git status command
        result = subprocess.run(['git', 'status', '--porcelain'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check the result
        if result.returncode != 0:
            print(f"Error checking git status: {result.stderr}")
            return False, False

        # Check for tracked and untracked changes
        tracked_changes = any(line[:2].strip() != '??' for line in result.stdout.splitlines())
        untracked_changes = any(line[:2] == '??' for line in result.stdout.splitlines())

        return tracked_changes, untracked_changes

    finally:
        # Change back to the original directory
        os.chdir(original_path)

if __name__ == '__main__':
    tracked_changes, untracked_changes = check_git_changes('.')
    if tracked_changes:
        print("There are tracked changes")
    else:
        print("There are no tracked changes")
        df_current = ResultDataProcessor().data
        last_commit = os.popen('git rev-parse HEAD').read().strip()
        print(last_commit)
        # save the current output to a file
        df_current.to_parquet(f'output_{last_commit}.parquet', index=True)
        print("Saved output to file")
    if untracked_changes:
        print("There are untracked changes")
    else:
        print("There are no untracked changes")

        
