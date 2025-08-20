import gc
from datetime import datetime
from pathlib import Path

from src.data_preprocessing.utils.save_last_run import save_last_run
from src.data_preprocessing.utils.bs_filtering import find_latest_file, filter_for_anomalous_candidates, \
    analyze_serial_number_book_states
from src.data_preprocessing.utils.dynamic_exclusion import DynamicExclusionPipeline
from src.data_preprocessing.utils.load_paths import load_last_run


def run_stage_1(input_file, base_filename, timestamp, stage_paths):
    unique_serials_csv = stage_paths["stage1_serials_csv"].format(
        base_filename=base_filename, timestamp=timestamp
    )
    unique_stations_csv = stage_paths["stage1_stations_csv"].format(
        base_filename=base_filename, timestamp=timestamp
    )

    filter_for_anomalous_candidates(
        input_file,
        target_book_state=2,
        unique_stations=unique_stations_csv,
        unique_serial_numbers=unique_serials_csv,
    )

    # Clean up memory
    gc.collect()

    return unique_serials_csv, unique_stations_csv


def run_stage_2(input_file, output_dir, base_filename, stage_paths):
    latest_serials_csv = find_latest_file(
        str(output_dir),
        prefix=f"{base_filename}_serial_numbers_for_filtering_",
        extension=".csv",
    )

    if not latest_serials_csv:
        raise FileNotFoundError("Could not find the serial number IDs CSV from Stage 1.")

    print(f"Found latest Stage 1 output: {latest_serials_csv}")

    summary_csv = stage_paths["stage2_summary_csv"].format(
        base_filename=base_filename, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    analyze_serial_number_book_states(
        input_file,
        latest_serials_csv,
        summary_csv,
    )

    # Clean up memory
    gc.collect()


def run_stage_3(input_file, output_dir, base_filename, stage_paths):
    stage2_summary_csv = find_latest_file(
        str(output_dir),
        prefix=f"{base_filename}_serial_number_book_state_summary_",
        extension=".csv",
    )

    if not stage2_summary_csv:
        raise FileNotFoundError("Could not find the serial number summary CSV from Stage 2.")

    print(f"Found latest Stage 2 output: {stage2_summary_csv}")

    exclusion_report = stage_paths["stage3_exclusion_report"].format(
        base_filename=base_filename, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    anomalous_serials = stage_paths["stage3_anomalous_csv"].format(
        base_filename=base_filename, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    final_filtered = stage_paths["stage3_final_filtered"].format(
        base_filename=base_filename, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    pipeline = DynamicExclusionPipeline(
        source_file=input_file,
        summary_csv=stage2_summary_csv,
        filtered_parquet=final_filtered,
        anomalies_csv=anomalous_serials,
        report_csv=exclusion_report
    )

    pipeline.run()

    # Clean up memory
    gc.collect()


def main():
    try:
        # --- INPUT PATH HANDLING ---
        print("Loading bookmeas dataset...")
        reuse_last = input("Reuse last input/output paths? (y/n): ").strip().lower() == "y"

        if reuse_last:
            last_paths = load_last_run()
            input_path = last_paths.get("input_path")
            output_path = last_paths.get("output_path")

            if not input_path or not output_path:
                print("No previous input/output paths found. Please enter them manually.")
                input_path = input("Enter path to input bookmeas dataset (parquet): ").strip()
                output_path = input("Enter output directory: ").strip()
        else:
            input_path = input("Enter path to input bookmeas dataset (parquet): ").strip()
            output_path = input("Enter output directory: ").strip()

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using input: {input_path}")
        print(f"Using output directory: {output_dir}")

        # --- CONFIGURE OUTPUT ---
        base_filename = input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / "bookmeas"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- SET STAGE PATHS ---
        stage_paths = {
            "stage1_serials_csv": str(output_dir / f"{base_filename}_serial_numbers_for_filtering_{timestamp}.csv"),
            "stage1_stations_csv": str(output_dir / f"{base_filename}_stations_for_filtering_{timestamp}.csv"),
            "stage2_summary_csv": str(output_dir / f"{base_filename}_serial_number_book_state_summary_{timestamp}.csv"),
            "stage3_exclusion_report": str(output_dir / f"{base_filename}_exclusion_report_{timestamp}.csv"),
            "stage3_anomalous_csv": str(output_dir / f"{base_filename}_anomalous_serial_numbers_{timestamp}.csv"),
            "stage3_final_filtered": str(output_dir / f"{base_filename}_final_filtered_{timestamp}.parquet"),
        }

        # Run pipeline stages
        run_stage_1(str(input_path), base_filename, timestamp, stage_paths)
        run_stage_2(str(input_path), output_dir, base_filename, stage_paths)
        run_stage_3(str(input_path), output_dir, base_filename, stage_paths)

        print("\n--- Pipeline execution completed successfully ---")
        print(f"All reports and final files are saved to: {str(output_dir)}")

        # --- SAVE PATHS TO CONFIG ---
        save_last_run(
            input_path=str(input_path),
            output_path=str(output_dir),
            version="v1"
        )

    except Exception as e:
        print(f"\n--- Pipeline failed: {e} ---")


if __name__ == "__main__":
    main()
