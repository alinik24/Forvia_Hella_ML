import os
import shutil
import tempfile
import traceback

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import compute as pc
from skrub import deduplicate


def validate_input_paths(materials_path: str, bookmeas_path: str) -> None:
    """Validate existence of input files"""
    if not os.path.exists(materials_path):
        raise FileNotFoundError(f"!Materials file not found: {materials_path}!")
    if not os.path.exists(bookmeas_path):
        raise FileNotFoundError(f"!Bookmeas file not found: {bookmeas_path}!")


def create_temp_output_path(output_path: str) -> str:
    """Create temporary output path to avoid partial writes"""
    output_dir = os.path.dirname(output_path) or "."
    return os.path.join(output_dir, f"TEMP_{os.path.basename(output_path)}")


def get_merged_schema(
        materials_ds: ds.Dataset,
        bookmeas_ds: ds.Dataset,
        serial_col: str
) -> pa.Schema:
    """Create merged schema from both datasets"""
    materials_schema = materials_ds.schema
    bookmeas_schema = bookmeas_ds.schema
    return pa.schema(
        list(materials_schema) +
        [f for f in bookmeas_schema if f.name != serial_col]
    )


def ensure_output_directory(output_path: str) -> None:
    """Create output directory if needed"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def process_batch(
        bookmeas_batch: pa.RecordBatch,
        materials_ds: ds.Dataset,
        writer: pq.ParquetWriter,
        merged_schema: pa.Schema,
        serial_col: str,
        batch_count: int
) -> int:
    """
    Process a single batch and return number of merged rows
    Returns -1 if batch is skipped due to error
    """
    batch_rows = bookmeas_batch.num_rows
    try:
        # Extract serial numbers from current bookmeas batch
        serials = bookmeas_batch[serial_col]

        # Filter materials for matching serials
        materials_batch = materials_ds.to_table(
            filter=pc.field(serial_col).isin(serials))

        # Skip if no matches
        if materials_batch.num_rows == 0:
            print(f"!Batch {batch_count}: No matching materials found for {batch_rows} bookmeas entries!")
        return 0

        # Convert to pandas for efficient merging
        materials_df = materials_batch.to_pandas()
        bookmeas_df = bookmeas_batch.to_pandas()

        # Merge batch data
        merged_df = materials_df.merge(
            bookmeas_df,
            on=serial_col,
            how="inner"
        )

        # Convert back to Arrow and write
        merged_batch = pa.Table.from_pandas(merged_df, schema=merged_schema)
        writer.write_table(merged_batch)
        return merged_batch.num_rows

    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, KeyError) as e:
        print(f"!DATA ERROR in batch {batch_count} ({batch_rows} rows): {str(e)}")
        return -1

    except MemoryError:
        print(f"!MEMORY ERROR in batch {batch_count} ({batch_rows} rows)")
        print("Consider reducing batch size or increasing system memory")
        return -1

    except Exception:
        print(f"!UNEXPECTED ERROR in batch {batch_count} ({batch_rows} rows):")
        print(traceback.format_exc())
        return -1


def handle_main_errors(error: Exception) -> None:
    """Handle errors in main processing flow"""
    if isinstance(error, (FileNotFoundError, OSError)):
        print("!I/O ERROR during merging:")
        print(str(error))
        print("Check file paths and permissions")
    elif isinstance(error, pa.ArrowInvalid):
        print("!ARROW DATA ERROR during merging:")
        print(str(error))
        print("Check schema compatibility and data formats")
    elif isinstance(error, MemoryError):
        print("!CRITICAL MEMORY ERROR!")
        print("Processing halted due to insufficient memory")
    else:
        print("!UNEXPECTED ERROR during merging:")
        print(traceback.format_exc())


def cleanup_resources(writer: pq.ParquetWriter, temp_path: str) -> None:
    """Ensure proper cleanup of resources"""
    try:
        if writer:
            writer.close()
    except:
        pass

    try:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    except:
        pass


def deduplicate_with_skrub(input_path: str, output_path: str, key_col: str, timestamp_col: str = None) -> pd.DataFrame:
    """
    Deduplicate dataset using skrub with proper index handling.
    Returns exact deduplication when possible, fuzzy when needed.
    """
    print(f"\nDeduplicating {os.path.basename(input_path)} using skrub...")

    # Read dataset into pandas DataFrame and reset index
    df = pd.read_parquet(input_path).reset_index(drop=True)
    original_rows = len(df)

    # First try exact deduplication
    exact_dedup = df.drop_duplicates(subset=[key_col], keep='first')
    if len(exact_dedup) == len(df):
        print("No duplicates found - using exact deduplication")
        deduped_df = exact_dedup
    else:
        print(f"Found {len(df) - len(exact_dedup)} exact duplicates")

        # Extract the key column for fuzzy deduplication
        key_series = df[key_col].astype(str)

        print("Performing fuzzy deduplication on serial numbers...")
        canonical_serials = deduplicate(
            key_series,
            ngram_range=(2, 4),
            analyzer='char_wb',
            linkage_method='average'
        )

        # Add canonical serials to DataFrame - ensure index alignment
        df = df.reset_index(drop=True)
        df['canonical_serial'] = pd.Series(canonical_serials.values, index=df.index)

        # If timestamp is available, use it to keep most recent
        if timestamp_col and timestamp_col in df.columns:
            df = df.sort_values(timestamp_col, ascending=False)
            deduped_df = df.drop_duplicates(subset=['canonical_serial'], keep='first')
        else:
            deduped_df = df.drop_duplicates(subset=['canonical_serial'], keep='first')

        # Remove temporary column
        deduped_df = deduped_df.drop(columns=['canonical_serial'])

    # Write deduplicated data
    deduped_df.to_parquet(output_path)
    print(f"Deduplicated from {original_rows} to {len(deduped_df)} rows")
    print(f"Duplicate reduction: {1 - len(deduped_df) / original_rows:.2%}")
    print(f"Saved to {output_path}")
    return deduped_df


def merge_balanced_datasets(
        materials_path: str,
        bookmeas_path: str,
        output_path: str,
        serial_col: str = "serial_number_id"
) -> None:
    """Safe and efficient dataset merge with proper deduplication"""
    # Create temp directory for deduplicated files
    temp_dir = tempfile.mkdtemp()
    temp_output = output_path + ".tmp"

    try:
        print("\n" + "=" * 50)
        print("Starting Dataset Merge with Deduplication")
        print("=" * 50)

        # Create deduplicated paths
        dedup_mat_path = os.path.join(temp_dir, "dedup_materials.parquet")
        dedup_book_path = os.path.join(temp_dir, "dedup_bookmeas.parquet")

        # Deduplicate both datasets
        print("\nDeduplicating materials dataset...")
        dedup_materials = deduplicate_with_skrub(
            materials_path, dedup_mat_path, serial_col, "created_at"
        )

        print("\nDeduplicating bookmeas dataset...")
        dedup_bookmeas = deduplicate_with_skrub(
            bookmeas_path, dedup_book_path, serial_col, "book_stamp"
        )

        # Diagnostic information
        print("\n" + "-" * 50)
        print("Dataset Diagnostics After Deduplication")
        print("-" * 50)
        print(f"Materials rows: {len(dedup_materials)}")
        print(f"Bookmeas rows: {len(dedup_bookmeas)}")

        # Sample serials comparison
        mat_sample = dedup_materials[serial_col].head(5).tolist()
        book_sample = dedup_bookmeas[serial_col].head(5).tolist()
        print(f"\nSample materials serials: {mat_sample}")
        print(f"Sample bookmeas serials: {book_sample}")

        # Check for common serials
        common_serials = set(dedup_materials[serial_col]).intersection(set(dedup_bookmeas[serial_col]))
        print(f"\nCommon serials between datasets: {len(common_serials)}")

        # Merge datasets
        print("\n" + "-" * 50)
        print("Starting Merge Process")
        print("-" * 50)

        # Perform merge
        merged_df = pd.merge(
            dedup_materials,
            dedup_bookmeas,
            on=serial_col,
            how="inner",
            suffixes=("_materials", "_bookmeas")
        )

        # Prepare output directory
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Write merged data
        merged_df.to_parquet(output_path)

        print("\n" + "=" * 50)
        print("Merge Complete")
        print("=" * 50)
        print(f"Successfully merged {len(merged_df)} rows to {output_path}")

        # Final validation
        unique_serials = merged_df[serial_col].nunique()
        print(f"Unique serials in merged data: {unique_serials}")
        print(f"Merge efficiency: {len(merged_df) / min(len(dedup_materials), len(dedup_bookmeas)):.2%}")

    except Exception as e:
        print(f"\nERROR during merge: {str(e)}")
        traceback.print_exc()
    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)