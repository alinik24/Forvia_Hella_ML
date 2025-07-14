import os
import traceback
from typing import Tuple, Set, Dict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def extract_serials_by_state(
    dataset_path: str,
    serial_col: str,
    state_col: str,
    allowed_serials: Set
) -> Tuple[Dict[int, Set], Dict[int, int]]:
    """
    Extracts serials grouped by class label (e.g., book_state), filtered to only serials in `allowed_serials`.
    Returns a dictionary {state: set of serials} and a count dictionary.
    """
    state_to_serials = {0: set(), 1: set(), 2: set()}
    state_counts = {0: 0, 1: 0, 2: 0}

    try:
        dataset = ds.dataset(dataset_path, format="parquet")
        allowed_array = pa.array(list(allowed_serials))

        for batch in dataset.to_batches(columns=[serial_col, state_col]):
            try:
                mask = pc.is_in(batch[serial_col], allowed_array)
                filtered = batch.filter(mask)
                if filtered.num_rows == 0:
                    continue

                for state in [0, 1, 2]:
                    state_mask = pc.equal(filtered[state_col], state)
                    serials = filtered[serial_col].filter(state_mask)
                    serial_list = serials.to_pylist()
                    state_to_serials[state].update(serial_list)
                    state_counts[state] += len(serial_list)

            except Exception as e:
                print(f"!Error during batch processing: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"!Failed to read dataset at {dataset_path}: {e}")
        traceback.print_exc()
        raise

    return state_to_serials, state_counts


def load_serials(path: str, serial_col: str) -> Set:
    """
    Load all unique serial numbers from a Parquet dataset column.
    """
    try:
        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table(columns=[serial_col])
        serials = []
        for chunk in table[serial_col].chunks:
            serials.extend(chunk.to_pylist())
        return set(serials)
    except Exception as e:
        print(f"!Failed to load serials from {path}: {e}")
        traceback.print_exc()
        raise



def sample_zero_serials_with_diversity(
    zero_serials: Set,
    target_size: int,
    random_state: int = 42
) -> Set:
    """
    Samples a subset of 0-state serials while preserving serial diversity.
    """
    if len(zero_serials) <= target_size:
        return zero_serials

    rng = np.random.default_rng(random_state)
    sampled = rng.choice(list(zero_serials), size=target_size, replace=False)
    return set(sampled)


def filter_and_write_dataset(
    input_path: str,
    output_path: str,
    serials_to_keep: Set,
    serial_col: str
) -> None:
    """
    Filters a Parquet dataset to rows where serial_col is in serials_to_keep and writes it out.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_path = output_path + ".tmp"
        serial_array = pa.array(list(serials_to_keep))

        dataset = ds.dataset(input_path, format="parquet")
        filtered_table = dataset.to_table(filter=pc.field(serial_col).isin(serial_array))

        pq.write_table(filtered_table, temp_path)
        os.replace(temp_path, output_path)
    except Exception as e:
        print(f"!Error writing filtered dataset: {e}")
        traceback.print_exc()
        raise
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def balance_by_state_per_serial(
    input_main_path: str,
    output_main_path: str,
    serial_col: str = "serial_number_id",
    bookstate_col: str = "book_state",
    zero_to_nonzero_ratio: int = 5,
    random_state: int = 42
) -> None:
    """
    Balances a large dataset by downsampling 0-class entries while retaining all 1 and 2 entries.
    Ensures serial diversity by not keeping only one serial.
    """
    try:
        print("Scanning input dataset...")
        dataset = ds.dataset(input_main_path, format="parquet")
        nonzero_serials: Set[str] = set()
        zero_serials_counts: Dict[str, int] = {}
        state_map: Dict[str, Set[int]] = {}

        # First pass: identify serials and bookstates
        for batch in dataset.to_batches(columns=[serial_col, bookstate_col]):
            serials = batch[serial_col]
            states = batch[bookstate_col]
            for s, b in zip(serials.to_pylist(), states.to_pylist()):
                if s not in state_map:
                    state_map[s] = set()
                state_map[s].add(b)

        # Split serials into non-zero vs only-zero
        for serial, states in state_map.items():
            if 1 in states or 2 in states:
                nonzero_serials.add(serial)
            elif states == {0}:
                zero_serials_counts[serial] = zero_serials_counts.get(serial, 0) + 1

        print(f"Found {len(nonzero_serials)} non-zero serials")
        print(f"Found {len(zero_serials_counts)} zero-only serials")

        # Sample from zero-only serials
        n_sample = min(len(zero_serials_counts), len(nonzero_serials) * zero_to_nonzero_ratio)
        sampled_zero_serials = set()
        if n_sample > 0:
            rng = np.random.default_rng(random_state)
            sampled_zero_serials = set(rng.choice(list(zero_serials_counts.keys()), size=n_sample, replace=False))
            print(f"Sampled {len(sampled_zero_serials)} zero-only serials")

        # Final serial list to keep
        serials_to_keep = nonzero_serials | sampled_zero_serials
        print(f"Total serials to retain: {len(serials_to_keep)}")

        # Filter and write dataset
        print("Filtering dataset...")
        serials_array = pa.array(list(serials_to_keep))
        filtered_table = dataset.to_table(filter=pc.field(serial_col).isin(serials_array))

        print("Writing output dataset...")
        os.makedirs(os.path.dirname(output_main_path), exist_ok=True)
        pq.write_table(filtered_table, output_main_path)
        print("Done. Balanced dataset saved.")

    except Exception as e:
        print(f"!UNEXPECTED ERROR during balancing: {str(e)}")
        traceback.print_exc()