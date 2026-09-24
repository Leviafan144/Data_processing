"""
Extract all data from Bio-Logic .mpr files and combine into single CSV.

Processes CV and PEIS files, handling different column structures.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from galvani import BioLogic
import matplotlib.pyplot as plt
import math
import decimal
import cv_analysis
decimal.getcontext().prec = 20


# Configuration
INPUT_DIR = Path(__file__).parent
DATA_DIR = INPUT_DIR / "data_row"
OUTPUT_CSV = INPUT_DIR / "combined_electrochemical_data.csv"
METRICS_CSV = INPUT_DIR / "data.csv"
LOG_FILE = INPUT_DIR / "extraction_log.txt"
METRICS_COLUMNS = ["source_file", "file_index", "loop_index",
                   "E1/2a", "Epa", "Ypa", "Spa",
                   "E1/2c", "Epc", "Ypc", "Spc"]


def setup_logging():
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def extract_technique(filename):
    """Extract technique type from filename (CV or PEIS)."""
    filename = str(filename).upper()
    if '_CV_' in filename:
        return 'CV'
    elif '_PEIS_' in filename:
        return 'PEIS'
    else:
        return 'UNKNOWN'

def read_mpr_file_safely(filepath):
    """
    Read .mpr file with error handling.

    Args:
        filepath: Path to .mpr file

    Returns:
        pandas.DataFrame or None if reading fails
    """
    try:
        logging.info(f"Reading: {filepath.name}")
        mpr = BioLogic.MPRfile(str(filepath))
        df = pd.DataFrame(mpr.data)

        if df.empty:
            logging.warning(f"Empty data in {filepath.name}")
            return None

        logging.info(f"Successfully read {filepath.name}: {len(df)} rows, {len(df.columns)} columns")
        return df

    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None

    except Exception as e:
        logging.error(f"Failed to read {filepath.name}: {type(e).__name__}: {e}")
        return None

def iter_mpr_files():
    """Рекурсивный поиск .mpr в DATA_DIR, пропуск служебных каталогов."""
    if not DATA_DIR.exists():
        return []
    files = []
    for p in DATA_DIR.rglob("*.mpr"):
        try:
            rel_parts = p.relative_to(DATA_DIR).parts
        except ValueError:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in rel_parts[:-1]):
            continue
        files.append(p)
    return sorted(files)


def main():
    """Main extraction workflow."""
    setup_logging()

    logging.info("=" * 60)
    logging.info("Bio-Logic .mpr File Extraction Started")
    logging.info(f"Input directory: {INPUT_DIR}")
    logging.info(f"Data directory: {DATA_DIR}")
    logging.info(f"Output CSV: {OUTPUT_CSV}")
    logging.info(f"Metrics CSV: {METRICS_CSV}")
    logging.info("=" * 60)

    # Find all .mpr files (recursive)
    mpr_files = iter_mpr_files()

    if not mpr_files:
        logging.error(f"No .mpr files found in {DATA_DIR}")
        return

    logging.info(f"Found {len(mpr_files)} .mpr files")

    # Read all files and collect dataframes
    all_dataframes = []
    successful_files = []
    failed_files = []
    all_metrics_rows = []


    def print_log(id, e12a, epa, ypa, spa, e12c, epc, ypc, spc):
        print("id =", id)
        print("     E1/2a =", e12a)
        print("     Epa =", epa)
        print("     Ypa =", ypa)
        print("     Spa =", spa)
        print("     E1/2c =", e12c)
        print("     Epc =", epc)
        print("     Ypc =", ypc)
        print("     Spc =", spc)

    for file_index, filepath in enumerate(mpr_files, start=1):
        try:
            source_id = filepath.relative_to(DATA_DIR).as_posix()
        except ValueError:
            source_id = filepath.name
        df = read_mpr_file_safely(filepath)
        if df is None:
            failed_files.append(source_id)
            continue

        ewe = df.get("Ewe/V")
        i_main = df.get("<I>/mA")

        #Проверка наличия колонок в файле
        if ewe is None or i_main is None:
            logging.error(f"Missing required columns in {filepath.name}")
            failed_files.append(source_id)
            continue

        del_x = decimal.Decimal("1")
        del_y = decimal.Decimal("1")

        cyc = df.get("cycle number")
        loops = []
        loop = []
        prev_cycle = cyc[0]
        for x in range(len(ewe)):
            if cyc[x] != prev_cycle:
                loops.append(loop)
                loop = []
                prev_cycle = cyc[x]
            loop.append((decimal.Decimal(str(ewe[x] * float(del_x))), decimal.Decimal(str(i_main[x])) * del_y))
        loops.append(loop)

        fig, axes = plt.subplots(nrows = max(1,math.ceil(len(loops)/5)), ncols=5)
        for loop_index in range(len(loops)):
            loop = loops[loop_index]
            x_loop = [loop[z][0] for z in range(len(loop))]
            x_max = max(x_loop)

            if loop_index == 1:
                plt.figure()
                plt.scatter(x_loop, [loop[z][1] for z in range(len(loop))], s = 1)

            axe = axes[loop_index//5, loop_index%5]
            axe.scatter(x_loop, [loop[z][1] for z in range(len(loop))], s = 1)
            axe.set_ylim(-1.4,1.4)
            axe.grid()

            e12a, epa, ypa, sa, e12c, epc, ypc, sc = cv_analysis.analyze_loop(
                loop, del_x, del_y, x_max, axe, loop_index)
            all_metrics_rows.append({
                "source_file": source_id,
                "file_index": file_index,
                "loop_index": loop_index,
                "E1/2a": e12a,
                "Epa": epa,
                "Ypa": ypa,
                "Spa": sa,
                "E1/2c": e12c,
                "Epc": epc,
                "Ypc": ypc,
                "Spc": sc,
            })
            print_log(loop_index, e12a, epa, ypa, sa, e12c, epc, ypc, sc)
        plt.show()


        # Add identification columns
        df.insert(0, 'source_file', source_id)
        df.insert(1, 'technique', extract_technique(filepath.name))
        df.insert(2, 'file_index', file_index)

        all_dataframes.append(df)
        successful_files.append((source_id, len(df)))

    # Export accumulated per-loop metrics once (all files)
    if all_metrics_rows:
        logging.info(f"Exporting metrics to CSV: {METRICS_CSV}")
        metrics_df = pd.DataFrame(all_metrics_rows, columns=METRICS_COLUMNS)
        metrics_df.to_csv(METRICS_CSV, index=False, sep=";")
    else:
        logging.warning("No CV metrics collected (empty loops?)")

    # Check if we have any data
    if not all_dataframes:
        logging.error("No data could be extracted from any files")
        return

    logging.info("="*60)
    logging.info(f"Successfully processed: {len(successful_files)}/{len(mpr_files)} files")

    # Combine all dataframes
    logging.info("Combining all dataframes...")
    combined_df = pd.concat(all_dataframes, axis=0, ignore_index=True)

    # Reorder columns for readability
    priority_cols = ['source_file', 'technique', 'file_index']

    # Add common electrochemical columns if they exist
    common_cols = ['time/s', 'cycle number', 'half cycle', 'Ewe/V', 'I/mA',
                   'control/V/mA', 'freq/Hz', '|Z|/Ohm', 'Phase(Z)/deg']

    for col in common_cols:
        if col in combined_df.columns and col not in priority_cols:
            priority_cols.append(col)

    # Add remaining columns alphabetically
    other_cols = sorted([c for c in combined_df.columns if c not in priority_cols])
    combined_df = combined_df[priority_cols + other_cols]

    # Export to CSV
    logging.info(f"Exporting to CSV: {OUTPUT_CSV}")
    combined_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    # Summary statistics
    logging.info("="*60)
    logging.info("EXTRACTION SUMMARY")
    logging.info("="*60)
    logging.info(f"Total rows extracted: {len(combined_df):,}")
    logging.info(f"Total columns: {len(combined_df.columns)}")
    logging.info(f"Output file size: {OUTPUT_CSV.stat().st_size / 1024:.2f} KB")
    logging.info("")

    logging.info("Successfully processed files:")
    for filename, row_count in successful_files:
        logging.info(f"  - {filename}: {row_count:,} rows")

    if failed_files:
        logging.warning("")
        logging.warning("Failed files:")
        for filename in failed_files:
            logging.warning(f"  - {filename}")

    logging.info("")
    logging.info("Column names:")
    for col in combined_df.columns:
        logging.info(f"  - {col}")

    logging.info("="*60)
    logging.info("Extraction completed successfully!")
    logging.info(f"Log file: {LOG_FILE}")
    logging.info("="*60)



if __name__ == "__main__":
    main()
