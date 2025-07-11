import os
import traceback
from typing import Tuple, List, Set, Dict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def load_serials_from_dataset(path: str, serial_col: str) -> Set:
    """Load all unique serial numbers from a dataset with error handling."""
    try:
        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table(columns=[serial_col])
        serials_list = []
        for chunk in table[serial_col].chunks:
            serials_list.extend(chunk.to_pylist())
        return set(serials_list)
    except (pa.ArrowInvalid, pa.ArrowIOError, OSError) as e:
        print(f"!ERROR loading serials from {path}: {str(e)}!")
        raise
    except Exception as e:
        print(f"!UNEXPECTED ERROR loading serials from {path}:!")
        traceback.print_exc()
        raise


def collect_critical_and_common_serials(
        bookmeas_path: str,
        materials_serials: Set,
        serial_col: str,
        bookstate_col: str
) -> Tuple[Set, List[pa.Array], Dict[int, int]]:
    """
    Collect serials with bookstate 1/2 (critical), all common serials,
    and bookstate distribution with error handling.
    """
    critical_serials = set()
    all_common_serials = []
    bookstate_counts = {0: 0, 1: 0, 2: 0}

    try:
        bookmeas_ds = ds.dataset(bookmeas_path, format="parquet")
        materials_array = pa.array(list(materials_serials))

        for batch in bookmeas_ds.to_batches(columns=[serial_col, bookstate_col]):
            try:
                # Create mask for common serials
                mask = pc.is_in(batch[serial_col], materials_array)
                filtered = batch.filter(mask)

                # Skip empty batches after filtering
                if filtered.num_rows == 0:
                    continue

                for bookstate in [1, 2]:
                    state_mask = pc.equal(filtered[bookstate_col], bookstate)
                    critical = filtered[serial_col].filter(state_mask)
                    # Convert to Python list safely
                    critical_list = critical.to_pylist()
                    critical_serials.update(critical_list)
                    bookstate_counts[bookstate] += len(critical_list)

                # Count zeros using Arrow compute
                zero_mask = pc.equal(filtered[bookstate_col], 0)
                bookstate_counts[0] += pc.sum(zero_mask).as_py()

                # Collect common serials
                all_common_serials.append(filtered[serial_col])

            except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as e:
                print(f"!BATCH PROCESSING ERROR: {str(e)}!")
                print("Skipping problematic batch")
            except MemoryError:
                print("!MEMORY ERROR IN BATCH PROCESSING!")
                print("Consider reducing batch size or increasing memory")
                raise
            except Exception:
                print("!UNEXPECTED BATCH ERROR:!")
                traceback.print_exc()

        return critical_serials, all_common_serials, bookstate_counts

    except (pa.ArrowInvalid, pa.ArrowIOError, OSError) as e:
        print(f"!ERROR processing bookmeas dataset: {str(e)}!")
        raise
    except Exception:
        print("!UNEXPECTED ERROR in serial collection:!")
        traceback.print_exc()
        raise


def sample_zero_serials(
        all_common_serials: List[pa.Array],
        critical_serials: Set,
        sample_size: int,
        random_state: int
) -> Set:
    """Sample a subset of serials from bookstate=0 entries with error handling."""
    try:
        # Handle empty case
        if not all_common_serials:
            return set()

        # Concatenate arrays safely
        all_serials_arr = pa.concat_arrays(all_common_serials)

        # Check for empty arrays
        if len(all_serials_arr) == 0:
            return set()

        # Create critical serials array
        critical_array = pa.array(list(critical_serials))

        # Create mask for non-critical serials
        in_mask = pc.is_in(all_serials_arr, critical_array)
        zero_mask = pc.invert(in_mask)
        zero_serials = all_serials_arr.filter(zero_mask)

        # Handle case with no zero serials
        if len(zero_serials) == 0:
            return set()

        # Handle case where sample size is larger than available
        available_zeros = len(zero_serials)
        if available_zeros < sample_size:
            print(f"!Warning: Only {available_zeros} zero-serials available, sampling all!")
            sample_size = available_zeros

        # Sample indices
        rng = np.random.default_rng(random_state)
        if sample_size > 0:
            sampled_indices = rng.choice(available_zeros, size=sample_size, replace=False)
            sampled_zero_serials = zero_serials.take(sampled_indices)
            return set(sampled_zero_serials.to_pylist())
        return set()

    except (pa.ArrowInvalid, pa.ArrowCapacityError) as e:
        print(f"!SAMPLING ERROR: {str(e)}!")
        raise
    except ValueError as ve:
        print(f"!VALUE ERROR IN SAMPLING: {str(ve)}!")
        raise
    except MemoryError:
        print("!MEMORY ERROR DURING SAMPLING!")
        raise
    except Exception:
        print("!UNEXPECTED SAMPLING ERROR:")
        traceback.print_exc()
        raise


def write_filtered_dataset(input_path: str, output_path: str, serials: Set, serial_col: str) -> None:
    """Write a dataset filtered by a set of serial numbers with error handling."""
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use temporary file for safe writes
        temp_path = output_path + ".tmp"
        serial_array = pa.array(list(serials))

        dataset = ds.dataset(input_path, format="parquet")
        filtered = dataset.to_table(filter=pc.field(serial_col).isin(serial_array))

        pq.write_table(filtered, temp_path)
        os.replace(temp_path, output_path)

    except (pa.ArrowIOError, OSError) as e:
        print(f"!I/O ERROR writing {output_path}: {str(e)}!")
        # Clean up temporary file if exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise
    except pa.ArrowInvalid as e:
        print(f"!ARROW ERROR writing {output_path}: {str(e)}!")
        raise
    except Exception:
        print(f"!UNEXPECTED ERROR writing {output_path}:")
        traceback.print_exc()
        raise
    finally:
        # Ensure temp file is cleaned up
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def balance_datasets(
        input_materials_path: str,
        input_bookmeas_path: str,
        output_materials_path: str,
        output_bookmeas_path: str,
        serial_col: str = "serial_number_id",
        bookstate_col: str = "book_state",
        factor: int = 5,
        random_state: int = 42
) -> None:
    """
    Balance bookmeas and materials datasets with comprehensive error handling.
    Preserves original logic while adding memory safety and error recovery.
    """
    try:
        # Validate input paths
        for path in [input_materials_path, input_bookmeas_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")

        print("Reading material serials...")
        materials_serials = load_serials_from_dataset(input_materials_path, serial_col)
        print(f"Found {len(materials_serials)} material serials")

        print("Scanning bookmeas to collect serials...")
        critical_serials, all_common_serials, bookstate_counts = collect_critical_and_common_serials(
            input_bookmeas_path, materials_serials, serial_col, bookstate_col
        )
        print(f"Found {len(critical_serials)} critical serials, {bookstate_counts[0]} zero entries")

        n_non_zero = len(critical_serials)
        n_zero = min(bookstate_counts[0], n_non_zero * factor) if n_non_zero > 0 else 0
        print(f"Balancing dataset: {n_non_zero} non-zero entries, sampling {n_zero} zeros")

        print("Sampling zero entries...")
        sampled_zero_serials = sample_zero_serials(
            all_common_serials, critical_serials, n_zero, random_state
        ) if n_zero > 0 else set()

        final_serials = critical_serials | sampled_zero_serials
        print(f"Total serials after balancing: {len(final_serials)}")

        print("Writing filtered materials...")
        write_filtered_dataset(input_materials_path, output_materials_path, final_serials, serial_col)

        print("Writing filtered bookmeas...")
        write_filtered_dataset(input_bookmeas_path, output_bookmeas_path, final_serials, serial_col)

        print("Balanced datasets written successfully.")

    except FileNotFoundError as fnfe:
        print(f"!CRITICAL ERROR: {str(fnfe)}!")
    except MemoryError:
        print("!MEMORY ERROR: Processing halted due to insufficient memory!")
    except (pa.ArrowInvalid, pa.ArrowIOError) as ae:
        print(f"!ARROW ERROR: {str(ae)}!")
    except Exception as e:
        print(f"!UNEXPECTED ERROR: {str(e)}!")
        traceback.print_exc()
