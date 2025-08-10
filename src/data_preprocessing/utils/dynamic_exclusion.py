import gc
import pandas as pd
import pyarrow.parquet as pq


class DynamicExclusionPipeline:
    def __init__(self, source_file, summary_csv,
                 filtered_parquet, anomalies_csv, report_csv):
        self.source_file = source_file
        self.summary_csv = summary_csv
        self.filtered_parquet = filtered_parquet
        self.anomalies_csv = anomalies_csv
        self.report_csv = report_csv

        self.df_summary = None
        self.df_source = None
        self.df_filtered = None
        self.anomalous_serials = set()

    def run(self):
        """
        Executes the entire dynamic exclusion pipeline.
        """
        print("\n=== Starting Stage 3: Dynamic Exclusion Pipeline ===")
        try:
            self._load_summary()
            self._detect_anomalies()
            self._load_source()
            self._filter_data()
            self._save_outputs()
            self._generate_report()
            print("=== Stage 3 completed successfully ===")
        except Exception as e:
            print(f"Error during pipeline: {e}")
            raise
        finally:
            self.df_summary = None
            self.df_source = None
            self.df_filtered = None
            gc.collect()

    def _load_summary(self):
        """
        Loads the summary CSV and prepares the 'serial_number_id' column.
        """
        print("Loading summary CSV...")
        self.df_summary = pd.read_csv(self.summary_csv, dtype={"Serial Number ID": "string"})
        self.df_summary.rename(columns={"Serial Number ID": "serial_number_id"}, inplace=True)
        print(f"Summary loaded with {len(self.df_summary)} rows.")

    def _detect_anomalies(self, col="Book State 1 Count"):
        """
        Identifies anomalous serial numbers using the Interquartile Range (IQR) method.
        """
        print("Detecting anomalies with IQR...")
        if col not in self.df_summary.columns:
            raise ValueError(f"Missing column '{col}' in summary CSV.")

        q1 = self.df_summary[col].quantile(0.25)
        q3 = self.df_summary[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}, Upper Bound: {upper_bound:.2f}")

        self.anomalous_serials = set(
            self.df_summary[self.df_summary[col] > upper_bound]["serial_number_id"]
        )
        print(f"Identified {len(self.anomalous_serials)} anomalous serials.")

    def _load_source(self):
        """
        Loads the source Parquet file, preserving all columns.
        """
        print("Loading source data...")
        # FIX: Load all columns by not specifying the 'columns' parameter.
        self.df_source = pd.read_parquet(self.source_file)
        
        # Ensure the key column for filtering is in the DataFrame.
        if "serial_number_id" not in self.df_source.columns:
            # Handle potential case where the column name might differ
            raise ValueError("The 'serial_number_id' column is missing from the source data.")
        
        self.df_source["serial_number_id"] = self.df_source["serial_number_id"].astype("string")
        # Ensure other key columns are handled gracefully if they exist
        if "station_desc" in self.df_source.columns:
            self.df_source["station_desc"] = self.df_source["station_desc"].astype("category")
        if "book_state" in self.df_source.columns:
            self.df_source["book_state"] = pd.to_numeric(self.df_source["book_state"], errors="coerce")

        print(f"Source data loaded with shape: {self.df_source.shape}")

    def _filter_data(self):
        """
        Filters the source data by excluding rows with anomalous serial numbers.
        """
        print(f"Filtering out {len(self.anomalous_serials)} anomalous serials...")
        self.df_filtered = self.df_source[~self.df_source["serial_number_id"].isin(self.anomalous_serials)]
        print(f"Filtered data shape: {self.df_filtered.shape}")

    def _save_outputs(self):
        """
        Saves the filtered data and the list of anomalous serials.
        """
        print("Saving outputs...")
        self.df_filtered.to_parquet(self.filtered_parquet, index=False)
        pd.DataFrame(list(self.anomalous_serials), columns=["Anomalous Serial Number ID"]).to_csv(
            self.anomalies_csv, index=False
        )
        print(f"Filtered data -> {self.filtered_parquet}")
        print(f"Anomalies list -> {self.anomalies_csv}")

    def _generate_report(self):
        """
        Generates a summary report of the exclusion process.
        """
        print("Generating report...")
        metrics = {
            "Metric": [],
            "Value": []
        }

        # Helper function to add metrics for a given DataFrame
        def add_metrics(label, df):
            metrics["Metric"].append(f"--- {label} ---")
            metrics["Value"].append("")
            metrics["Metric"] += ["Total Rows", "Unique serial number IDs", "Unique station_desc"]
            metrics["Value"] += [len(df), df["serial_number_id"].nunique(), df["station_desc"].nunique() if "station_desc" in df.columns else "N/A"]
            
            if "book_state" in df.columns:
                counts = df["book_state"].value_counts().to_dict()
                for state in sorted(counts.keys()):
                    metrics["Metric"].append(f"Book State {int(state)} Row Count ({label})")
                    metrics["Value"].append(counts[state])
                
                s1 = set(df[df["book_state"] == 1]["serial_number_id"])
                s2 = set(df[df["book_state"] == 2]["serial_number_id"])

                metrics["Metric"] += [
                    f"Total Unique IDs with either book state 1 or 2 ({label})",
                    f"Total Unique IDs with both book state 1 and 2 ({label})",
                    f"Total Unique IDs with only book state 1 ({label})"
                ]
                metrics["Value"] += [
                    len(s1.union(s2)),
                    len(s1.intersection(s2)),
                    len(s1.difference(s2))
                ]

        add_metrics("Source Data (Before Exclusion)", self.df_source)
        add_metrics("Filtered Data (After Exclusion)", self.df_filtered)

        # Final summary
        metrics["Metric"].append("--- Summary of Exclusions ---")
        metrics["Value"].append("")
        metrics["Metric"].append("Total Rows Eliminated")
        metrics["Value"].append(len(self.df_source) - len(self.df_filtered))
        metrics["Metric"].append("Serial Number IDs Eliminated")
        metrics["Value"].append(
            self.df_source["serial_number_id"].nunique() - self.df_filtered["serial_number_id"].nunique()
        )

        pd.DataFrame(metrics).to_csv(self.report_csv, index=False)
        print(f"Report -> {self.report_csv}")