import pandas as pd


def separate_excel(file_path):
    """
    Separate the excel file into multiple files based on 1000 rows.
    """
    df = pd.read_excel(file_path)
    rows_per_file = 1000
    n_chunks = len(df) // rows_per_file
    if(n_chunks > rows_per_file):
        for i in range(n_chunks):
            start = i*rows_per_file
        stop = (i+1) * rows_per_file
        sub_df = df.iloc[start:stop]
        sub_df.to_excel(f"/output/path/to/test-{i}.xlsx", sheet_name="a")
        if stop < len(df):
            sub_df = df.iloc[stop:]
            sub_df.to_excel(f"/output/path/to/test-{i}.xlsx", sheet_name="a")
