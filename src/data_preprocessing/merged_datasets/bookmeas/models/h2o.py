import h2o
import os
from src.data_preprocessing.utils.model_utils.h2o_utils import run_automl_pipeline, save_last_run, load_last_run

# --- Main script execution ---

def main():
    """
    Main function to orchestrate the H2O AutoML pipeline.
    It handles user interaction for file paths, runs the ML pipeline,
    and manages the H2O cluster.
    """
    try:
        # Start H2O cluster
        print("Starting H2O cluster...")
        h2o.init(max_mem_size_GB=24)
        print("H2O cluster started.")

        # --- Handle input and output paths ---
        
        # Prompt user for input file path, with option to reuse
        print("\nLoading dataset...")
        reuse_last = input("Reuse last input path? (y/n): ").strip().lower() == "y"
        if reuse_last:
            last_paths = load_last_run()
            input_path = last_paths.get("input_path")
            if not input_path or not os.path.exists(input_path):
                print("No last input path found or path is invalid. Please enter manually.")
                input_path = input("Enter path to encoded bookmeas dataset (parquet): ").strip()
        else:
            input_path = input("Enter path to encoded bookmeas dataset (parquet): ").strip()

        print(f"Loading dataset from {input_path} ...")
        
        # Prompt user for output path for plots
        output_path = input("Enter a directory to save the plots (e.g., 'plots/'): ").strip()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created directory: {output_path}")

        # Save the paths for future runs
        save_last_run(input_path=input_path, output_path=output_path, version="v1")

        # --- Run the AutoML pipeline ---
        run_automl_pipeline(input_path, output_path)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Shutdown H2O cluster
        conn = h2o.connection()
        if conn and conn.connected:
            print("\nShutting down H2O cluster...")
            h2o.shutdown(prompt=False)

if __name__ == "__main__":
    main()

