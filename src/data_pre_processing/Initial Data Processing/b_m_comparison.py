"""
Script: compare_bookings_vs_materials.py

Purpose:
--------
This script compares the `bookings.parquet` and `materials.parquet` files in terms of:
- Shared columns and matching data types.
- Sampled row-wise similarity.
- Statistical comparison of data distributions (numeric and categorical).
- Missing value patterns.

Core Features:
--------------
- Extracts schema and metadata for both files.
- Identifies shared columns and compares data types.
- Samples up to 500 rows from each file and calculates row-level similarity on shared columns.
- Performs Kolmogorov–Smirnov tests for numeric columns and Chi-squared tests for categorical columns.
- Visualizes column distributions and saves plots.
- Compares missing value percentages per shared column.

Output:
-------
- `bookings_materials_extended_comparison.csv`: Sample-wise similarity results.
- `distribution_comparison.csv`: Statistical comparison results for each shared column.
- `missing_values_comparison.csv`: Percentage of missing values per shared column.
- `plots/distribution_<column>.png`: Histograms for each comparable column.

Usage:
------
Just run the script after configuring the input/output paths. Ensure the two Parquet files exist and contain compatible data.

Suitable for:
-------------
- Pre-integration schema/data compatibility checks.
- Detecting redundancy or divergence between tables in production datasets.
"""
import pyarrow.parquet as pq
import pandas as pd
import os
import random
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
input_path = 'C:/Users/alina/Downloads/data_hella/'
output_dir = 'C:/Users/alina/OneDrive/Desktop/RWML projects/data_hella/output/'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'bookings_materials_extended_comparison.csv')
output_plots_dir = os.path.join(output_dir, 'plots/')
os.makedirs(output_plots_dir, exist_ok=True)

# Files to compare
files = ['bookings.parquet', 'materials.parquet']

# Function to get metadata (columns, types, row count)
def get_parquet_metadata(file_path):
    try:
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema_arrow
        columns = [(field.name, str(field.type)) for field in schema]
        total_rows = parquet_file.metadata.num_rows
        metadata = {'compression': parquet_file.metadata.row_group(0).column(0).compression}
        return columns, total_rows, metadata, None
    except Exception as e:
        return None, None, None, str(e)

# Function to sample a specific row (unchanged)
def sample_row(file_path, row_idx, total_rows):
    try:
        parquet_file = pq.ParquetFile(file_path)
        if row_idx >= total_rows:
            return None, f"Row index {row_idx} exceeds total rows {total_rows}"
        row_group_idx = 0
        cumulative_rows = 0
        for i in range(parquet_file.num_row_groups):
            rows_in_group = parquet_file.metadata.row_group(i).num_rows
            if cumulative_rows + rows_in_group > row_idx:
                row_group_idx = i
                break
            cumulative_rows += rows_in_group
        row_group = parquet_file.read_row_group(row_group_idx)
        local_idx = row_idx - cumulative_rows
        df = row_group.slice(local_idx, 1).to_pandas()
        return df.iloc[0], None
    except Exception as e:
        return None, str(e)

# Function to compare rows (unchanged)
def compare_rows(row1, row2, shared_columns):
    if shared_columns:
        matches = sum(1 for col in shared_columns if row1[col] == row2[col] or (pd.isna(row1[col]) and pd.isna(row2[col])))
        similarity = matches / len(shared_columns)
    else:
        types1 = [type(v).__name__ for v in row1.values]
        types2 = [type(v).__name__ for v in row2.values]
        type_matches = sum(1 for t1, t2 in zip(types1, types2) if t1 == t2)
        similarity = type_matches / max(len(types1), len(types2))
    return similarity

# Function to compare distributions
def compare_distributions(df1, df2, shared_columns):
    results = []
    for col in shared_columns:
        if df1[col].dtype in ['int64', 'float64'] and df2[col].dtype in ['int64', 'float64']:
            # Numeric: KS test
            ks_stat, p_value = stats.ks_2samp(df1[col].dropna(), df2[col].dropna())
            results.append({
                'column': col,
                'type': 'numeric',
                'ks_stat': ks_stat,
                'p_value': p_value,
                'interpretation': 'Similar' if p_value > 0.05 else 'Different'
            })
            # Plot distribution
            plt.figure(figsize=(8, 6))
            sns.histplot(df1[col].dropna(), label='bookings.parquet', color='blue', alpha=0.5)
            sns.histplot(df2[col].dropna(), label='materials.parquet', color='red', alpha=0.5)
            plt.title(f'Distribution of {col}')
            plt.legend()
            plt.savefig(os.path.join(output_plots_dir, f'distribution_{col}.png'))
            plt.close()
        elif df1[col].dtype == 'object' or df2[col].dtype == 'object':
            # Categorical: Chi-squared test
            freq1 = df1[col].value_counts()
            freq2 = df2[col].value_counts()
            common_cats = freq1.index.intersection(freq2.index)
            if len(common_cats) > 0:
                obs = np.array([freq1[common_cats], freq2[common_cats]])
                chi2, p_value, _, _ = stats.chi2_contingency(obs)
                results.append({
                    'column': col,
                    'type': 'categorical',
                    'chi2_stat': chi2,
                    'p_value': p_value,
                    'interpretation': 'Similar' if p_value > 0.05 else 'Different'
                })
    return results

# Function to check missing values
def check_missing_values(df1, df2, shared_columns):
    results = []
    for col in shared_columns:
        missing1 = df1[col].isna().mean() * 100
        missing2 = df2[col].isna().mean() * 100
        results.append({
            'column': col,
            'missing_bookings_percent': missing1,
            'missing_materials_percent': missing2,
            'difference': abs(missing1 - missing2)
        })
    return results

# Initialize results and metadata
results = []
column_comparison = {}

# Get metadata for both files
for file in files:
    file_path = os.path.join(input_path, file)
    columns, total_rows, metadata, error = get_parquet_metadata(file_path)
    if error:
        print(f"Error reading {file}: {error}")
        results.append({
            'row_index': None,
            'file1': file,
            'file2': files[1] if file == files[0] else files[0],
            'shared_columns': None,
            'similarity_score': None,
            'error': error
        })
        continue
    column_comparison[file] = {'columns': columns, 'total_rows': total_rows, 'metadata': metadata}

# Compare column structure, sample rows, and perform extended analyses
if all(file in column_comparison for file in files):
    cols1 = set(col[0] for col in column_comparison[files[0]]['columns'])
    cols2 = set(col[0] for col in column_comparison[files[1]]['columns'])
    shared_columns = cols1.intersection(cols2)
    shared_types = [
        col for col in shared_columns
        if dict(column_comparison[files[0]]['columns']).get(col) == dict(column_comparison[files[1]]['columns']).get(col)
    ]
    print(f"Shared columns: {shared_columns}")
    print(f"Shared columns with same type: {shared_types}")
    print(f"Metadata comparison: {column_comparison[files[0]]['metadata']} vs {column_comparison[files[1]]['metadata']}")
    
    # Load full datasets for distribution and missing value analysis
    df1 = pd.read_parquet(os.path.join(input_path, files[0]))
    df2 = pd.read_parquet(os.path.join(input_path, files[1]))
    
    # Compare distributions
    dist_results = compare_distributions(df1, df2, shared_columns)
    print("\nDistribution Comparison Results:")
    for res in dist_results:
        print(f"Column {res['column']} ({res['type']}): {res['interpretation']} (p-value = {res['p_value']:.4f})")
    
    # Check missing values
    missing_results = check_missing_values(df1, df2, shared_columns)
    print("\nMissing Values Comparison:")
    for res in missing_results:
        print(f"Column {res['column']}: Bookings {res['missing_bookings_percent']:.2f}% vs Materials {res['missing_materials_percent']:.2f}% (Diff: {res['difference']:.2f}%)")
    
    # Sample 500 rows (increased from 50)
    max_rows = min(column_comparison[files[0]]['total_rows'], column_comparison[files[1]]['total_rows'])
    if max_rows < 1:
        print("One or both files have no rows.")
        results.append({
            'row_index': None,
            'file1': files[0],
            'file2': files[1],
            'shared_columns': len(shared_columns),
            'similarity_score': None,
            'error': "One or both files have no rows"
        })
    else:
        row_indices = random.sample(range(max_rows), min(500, max_rows))
        for i, row_idx in enumerate(row_indices, 1):
            row1, error1 = sample_row(os.path.join(input_path, files[0]), row_idx, column_comparison[files[0]]['total_rows'])
            row2, error2 = sample_row(os.path.join(input_path, files[1]), row_idx, column_comparison[files[1]]['total_rows'])
            if error1 or error2:
                error = error1 or error2
                print(f"Sample {i} (row {row_idx}): Error - {error}")
                results.append({
                    'row_index': row_idx,
                    'file1': files[0],
                    'file2': files[1],
                    'shared_columns': len(shared_columns),
                    'similarity_score': None,
                    'error': error
                })
                continue
            similarity = compare_rows(row1, row2, shared_columns)
            print(f"Sample {i} (row {row_idx}): Similarity score = {similarity:.2f}")
            results.append({
                'row_index': row_idx,
                'file1': files[0],
                'file2': files[1],
                'shared_columns': len(shared_columns),
                'similarity_score': similarity,
                'error': None
            })

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_csv, index=False)
print(f"\nSaved comparison results to {output_csv}")

# Save distribution and missing value results to CSV
pd.DataFrame(dist_results).to_csv(os.path.join(output_dir, 'distribution_comparison.csv'), index=False)
pd.DataFrame(missing_results).to_csv(os.path.join(output_dir, 'missing_values_comparison.csv'), index=False)
print(f"Saved distribution comparison to {output_dir}distribution_comparison.csv")
print(f"Saved missing values comparison to {output_dir}missing_values_comparison.csv")

# Summarize similarity
if results and any(r['similarity_score'] is not None for r in results):
    avg_similarity = np.mean([r['similarity_score'] for r in results if r['similarity_score'] is not None])
    print(f"\nAverage similarity score: {avg_similarity:.2f}")
    if shared_columns:
        print("Files have shared columns, suggesting structural similarity.")
        if avg_similarity > 0.5:
            print("High average similarity in shared columns; files likely have similar data patterns.")
            if all(res['interpretation'] == 'Similar' for res in dist_results):
                print("Statistical tests confirm similar distributions in shared columns.")
            if all(res['difference'] < 5 for res in missing_results):
                print("Similar missing value patterns reinforce file similarity.")
        else:
            print("Low similarity in shared columns; files likely represent different data.")
    else:
        print("No shared columns; files likely represent different data.")
        if avg_similarity > 0.5:
            print("Despite no shared columns, some data type similarity exists.")
else:
    print("\nNo successful comparisons; check file accessibility or errors.")