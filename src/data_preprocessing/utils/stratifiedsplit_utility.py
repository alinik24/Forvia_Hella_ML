# src/data_preprocessing/utils/stratifiedsplit_utility.py

import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from typing import Tuple
from datetime import datetime
import pytz
import numpy as np

def stratified_split_and_save(
    ddf: dd.DataFrame,
    output_dir: str,
    target_column: str,
    time_column: str = 'created_at',
    test_ratio: float = 0.1,  # Default to 10%
    original_filename: str = 'data'
):
    """
    Splits a Dask DataFrame into training and testing sets by finding a unique
    time-based cutoff for each class to achieve the desired test ratio.
    """
    print(f"Preparing time-based split to allocate exactly {100*test_ratio:.0f}% of each class to test set...")

    if time_column not in ddf.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame columns: {ddf.columns.tolist()}")

    total_rows = len(ddf)
    if total_rows == 0:
        print("Input DataFrame is empty. Aborting.")
        return

    # Compute class counts for ALL classes
    class_counts = ddf[target_column].value_counts().compute()
    classes = class_counts.index.to_list()
    if not classes:
        print("No classes found in the target column. Aborting.")
        return

    # Calculate individual test set size and cutoff time for each class
    test_rows_list = []
    train_rows_list = []
    class_cutoffs = {}

    for c in classes:
        total_class_count = class_counts[c]
        if total_class_count == 0:
            continue

        # 1. Calculate Test Size for Each Class
        n_test = int(total_class_count * test_ratio)
        n_train = total_class_count - n_test

        if n_test == 0:
            print(f"Class {c} has too few instances to create a test set. All instances assigned to training.")
            class_cutoffs[c] = None
            train_rows_list.append(ddf[ddf[target_column] == c])
            continue
        
        print(f"Class {c}: Total={total_class_count}, Desired Test Count={n_test}, Desired Train Count={n_train}")

        # Get all instances for the class and sort by time
        class_ddf = ddf[ddf[target_column] == c].sort_values(time_column)

        # 2. Find the Time Cutoff for Each Class
        # The cutoff is the timestamp of the (total_count - n_test)th instance
        # Dask `iloc` is not efficient, so we compute and use pandas `iloc`
        cutoff_index = n_train
        if cutoff_index >= len(class_ddf): # Handle cases where n_train is larger than class size
             cutoff_index = len(class_ddf) - 1
             print(f"Warning: n_train for Class {c} exceeds class size. Using last instance as cutoff.")
        
        # Get the timestamp for the cutoff point
        class_cutoffs[c] = class_ddf.head(cutoff_index + 1).tail(1)[time_column].iloc[0]

        # 3. Construct the Test Set
        # Instances at or after the cutoff time for this class
        test_rows_list.append(class_ddf[class_ddf[time_column] >= class_cutoffs[c]])

        # 4. Construct the Training Set
        # Instances before the cutoff time for this class
        train_rows_list.append(class_ddf[class_ddf[time_column] < class_cutoffs[c]])

    # Combine the partial DataFrames into final train and test sets
    train_ddf = dd.concat(train_rows_list, axis=0, interleave_partitions=True)
    test_ddf = dd.concat(test_rows_list, axis=0, interleave_partitions=True)

    # Preserve column order
    original_columns = ddf.columns.tolist()
    train_ddf = train_ddf[original_columns]
    test_ddf = test_ddf[original_columns]

    # Compute sizes and distributions for the report
    train_size = len(train_ddf)
    test_size = len(test_ddf)
    actual_train_ratio = train_size / total_rows if total_rows > 0 else 0
    actual_test_ratio = test_size / total_rows if total_rows > 0 else 0

    train_class_counts = train_ddf[target_column].value_counts().compute().reindex(classes).fillna(0).astype(int)
    test_class_counts = test_ddf[target_column].value_counts().compute().reindex(classes).fillna(0).astype(int)
    
    original_dist = (class_counts / total_rows).round(4)
    train_dist = (train_class_counts / train_size if train_size > 0 else pd.Series(0, index=classes)).round(4)
    test_dist = (test_class_counts / test_size if test_size > 0 else pd.Series(0, index=classes)).round(4)
    
    # Time ranges
    original_time_range = (ddf[time_column].min().compute(), ddf[time_column].max().compute())
    train_time_range = (train_ddf[time_column].min().compute(), train_ddf[time_column].max().compute()) if train_size > 0 else (None, None)
    test_time_range = (test_ddf[time_column].min().compute(), test_ddf[time_column].max().compute()) if test_size > 0 else (None, None)

    # Generate report DataFrame
    report_rows = [
        {"Metric": "Total rows", "Value": total_rows},
        {"Metric": "Train rows", "Value": f"{train_size} ({actual_train_ratio:.2%})"},
        {"Metric": "Test rows", "Value": f"{test_size} ({actual_test_ratio:.2%})"},
        {"Metric": "Original time range", "Value": f"{original_time_range}"},
        {"Metric": "Train time range", "Value": f"{train_time_range}"},
        {"Metric": "Test time range", "Value": f"{test_time_range}"},
    ]

    for c in classes:
        # Added original class count to the report
        report_rows.append({"Metric": f"Class {c} count (original)", "Value": class_counts[c]})
        
        cutoff_val = str(class_cutoffs.get(c, "N/A"))
        report_rows.append({"Metric": f"Class {c} cutoff", "Value": cutoff_val})
        report_rows.append({"Metric": f"Class {c} count (train)", "Value": train_class_counts[c]})
        report_rows.append({"Metric": f"Class {c} count (test)", "Value": test_class_counts[c]})
        report_rows.append({"Metric": f"Class {c} dist (orig/train/test)", "Value": f"{original_dist[c]}/{train_dist[c]}/{test_dist[c]}"})

    report_df = pd.DataFrame(report_rows)

    # Generate and save report with sample rows
    current_time = datetime.now(pytz.UTC).strftime('%Y%m%d')
    report_path = Path(output_dir) / f"{original_filename}_split_report_{current_time}.csv"

    # Get 10 sample rows from each set (optional, can be commented out for large datasets)
    original_sample = ddf.head(10)
    train_sample = train_ddf.head(10)
    test_sample = test_ddf.head(10)

    original_sample["__dataset__"] = "original"
    train_sample["__dataset__"] = "train"
    test_sample["__dataset__"] = "test"
    samples_df = pd.concat([original_sample, train_sample, test_sample], ignore_index=True)

    with open(report_path, "w", encoding="utf-8") as f:
        report_df.to_csv(f, index=False)
        f.write("\n\n--- SAMPLE ROWS ---\n")
        samples_df.to_csv(f, index=False)

    print(f"📄 Report & sample rows saved to: {report_path}")

    # Save the datasets
    train_filename = f"{original_filename}_train_{current_time}.parquet"
    test_filename = f"{original_filename}_test_{current_time}.parquet"
    train_path = Path(output_dir) / train_filename
    test_path = Path(output_dir) / test_filename

    # Repartition to a single file for efficient saving
    train_ddf = train_ddf.repartition(npartitions=1)
    test_ddf = test_ddf.repartition(npartitions=1)

    train_ddf.to_parquet(train_path, engine='pyarrow', compression='snappy', write_index=False)
    test_ddf.to_parquet(test_path, engine='pyarrow', compression='snappy', write_index=False)

    print(f"✅ Saved training data to: {train_path}")
    print(f"✅ Saved testing data to: {test_path}")