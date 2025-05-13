import pandas as pd
import os # For os.path.exists

def pandas_parse_utf16le_csv(file_path):
    """
    Reads a UTF-16 LE encoded CSV file (without BOM) directly using pandas.read_csv.
    """
    print(f"\n--- Pandas Direct UTF-16-LE Parsing: {file_path} ---")
    try:
        # Specify 'utf-16-le'
        df = pd.read_csv(file_path, encoding='utf-16-le', sep=',')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during pandas direct parsing with 'utf-16-le': {e}")
        # Fallback: try with python engine if the C engine fails with utf-16-le for some reason
        try:
            print("Trying pandas with 'utf-16-le' and engine='python'")
            df = pd.read_csv(file_path, encoding='utf-16-le', sep=',', engine='python')
            return df
        except Exception as e_py:
            print(f"Pandas direct parsing with 'utf-16-le' and python engine also failed: {e_py}")
            return pd.DataFrame()

# --- Example Usage with your problematic file (Pandas Direct Method) ---
problematic_file_path_pd = './data/Definite MG/2023-11-20 은행순/은행순 MG_Horizontal Saccade  B (1Hz).csv' # Ensure this is the exact path

if os.path.exists(problematic_file_path_pd):
    print("\nAttempting Pandas Direct Parsing Method...")
    df_pandas_direct = pandas_parse_utf16le_csv(problematic_file_path_pd)
    if not df_pandas_direct.empty:
        print("\n--- Pandas Direct Parsed DataFrame (df_pandas_direct) ---")
        print("df_pandas_direct.head():")
        print(df_pandas_direct.head())
        print("\ndf_pandas_direct.dtypes:")
        print(df_pandas_direct.dtypes)
        print("\ndf_pandas_direct.isnull().sum():")
        print(df_pandas_direct.isnull().sum())
        if 'LH' in df_pandas_direct.columns:
            try:
                mean_lh_pd = df_pandas_direct['LH'].mean()
                print(f"\nMean of 'LH' column (pandas direct): {mean_lh_pd}")
            except TypeError as te:
                 print(f"\nError calculating mean for 'LH' (pandas direct): {te}.")
    else:
        print("Pandas direct parsing resulted in an empty DataFrame.")
else:
    print(f"Error: The specific file '{problematic_file_path_pd}' was not found for pandas direct parsing.")
