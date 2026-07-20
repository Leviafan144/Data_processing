#!/usr/bin/env python3
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
decimal.getcontext().prec = 20


def linear_fit_decimal(x, y):
    n = len(x)
    if n < 2:
        return decimal.Decimal(0), decimal.Decimal(0), decimal.Decimal('-Infinity')

    n_d = decimal.Decimal(n)
    x_mean = sum(x) / n_d
    y_mean = sum(y) / n_d

    dx = [xi - x_mean for xi in x]
    dy = [yi - y_mean for yi in y]

    denom = sum(dxi ** 2 for dxi in dx)
    if denom == 0:
        return decimal.Decimal(0), y_mean, decimal.Decimal('-Infinity')

    k = sum(dxi * dyi for dxi, dyi in zip(dx, dy)) / denom
    b = y_mean - k * x_mean

    y_pred = [k * xi + b for xi in x]
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)

    if ss_tot == 0:
        r2 = decimal.Decimal(1) if ss_res == 0 else decimal.Decimal('-Infinity')
    else:
        r2 = decimal.Decimal(1) - ss_res / ss_tot

    return k, b, r2


def find_longest_linear_segment(x, y, min_points=10, r2_threshold=decimal.Decimal('0.999')):
    n = len(x)

    for length in range(n, min_points - 1, -1):
        for i in range(0, n - length + 1):
            j = i + length
            k, b, r2 = linear_fit_decimal(x[i:j], y[i:j])
            if r2 >= r2_threshold:
                return (i, j, k, b, r2)
    return None

# Configuration
INPUT_DIR = Path(__file__).parent
OUTPUT_CSV = INPUT_DIR / "combined_electrochemical_data.csv"
LOG_FILE = INPUT_DIR / "extraction_log.txt"


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

def simpson_integral(x, y):
    n = len(x) - 1
    if n < 2:
        return decimal.Decimal(0)
    if n % 2 == 1:
        n -= 1
    h = (x[n] - x[0]) / decimal.Decimal(n)
    result = y[0] + y[n]
    for i in range(1, n):
        result += decimal.Decimal(4) * y[i] if i % 2 == 1 else decimal.Decimal(2) * y[i]
    return result * h / decimal.Decimal(3)

def main():
    """Main extraction workflow."""
    setup_logging()

    logging.info("="*60)
    logging.info("Bio-Logic .mpr File Extraction Started")
    logging.info(f"Input directory: {INPUT_DIR}")
    logging.info(f"Output CSV: {OUTPUT_CSV}")
    logging.info("="*60)

    # Find all .mpr files
    mpr_files = sorted(INPUT_DIR.glob("*.mpr"))

    if not mpr_files:
        logging.error(f"No .mpr files found in {INPUT_DIR}")
        return

    logging.info(f"Found {len(mpr_files)} .mpr files")

    # Read all files and collect dataframes
    all_dataframes = []
    successful_files = []
    failed_files = []


    def closest_point(i_start, i_finish, loop, point1, point2):
        x1,y1 = point1
        x2,y2 = point2
        s_min = 10 ** 10
        p = loop[i_start]
        for z in range(i_start, i_finish):
            if x1 == x2:
                s = abs(loop[z][1] - y1)
            else:
                s = math.dist(loop[z], (loop[z][0], (y2 - y1) * (loop[z][0] - x1) / (x2 - x1) + y1))
            if s < s_min:
                s_min = s
                p = loop[z]
        return p

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

    def find_y_peak(kind, y_loop, k, b, x_loop):
        y_del_max = 0
        y_peak = y_loop[0]
        for i in range(0, len(y_loop)):
            if kind == "a":
                y_del = y_loop[i] - (k * x_loop[i] + b)
            else:
                y_del = (k * x_loop[i] + b) - y_loop[i]
            if y_del > y_del_max:
                y_peak = y_loop[i]
                y_del_max = y_del

        if y_del_max <= 0:
            logging.warning(f"Peak not found for kind={kind}")

        return y_peak

    def draw(panel, x1, x2, x_max, y1, y2, ya, kind, x_peak, y_peak, k, b, y12, p3):
        if kind == "a":
            color = "green"
        else:
            color = "red"
        panel.plot([x1, x2, x_max], [y1, y2, ya], color=color)
        panel.plot([x_peak, x_peak], [y_peak, k * x_peak + b], color=color)
        panel.plot([x1, x2, x_peak, x_max], [y1 + y12, y2 + y12, k * x_peak + b + y12, ya + y12], color=color)
        panel.scatter(x1, y1, color="black", s=7)
        panel.scatter(x2, y2, color="black", s=7)
        panel.scatter(x_peak, y_peak - y12, color="black", s=7)
        panel.scatter(p3[0], p3[1], color="black", s=7)

    def find_square(x_loop, y_loop, k, b):

        s = 0
        for i in range(len(x_loop)-1):
            x1 = x_loop[i]
            y1 = y_loop[i]
            x2 = x_loop[i+1]
            y2 = y_loop[i+1]

            y3 = k * x1 + b
            y4 = k * x2 + b

            s += (y1-y3+y2-y4) * (x2-x1) / 2

        return s

    def Chart1(loop,  del_x, del_y, axe, loop_num):

        def counting(loop, x_loop, y_loop, x_end, loop_num, kind):
            try:
                if kind == "a":
                    peak_idx = y_loop.index(max(y_loop))
                else:
                    peak_idx = y_loop.index(min(y_loop))
                start, end, k, b, r2 = find_longest_linear_segment(x_loop[:peak_idx], y_loop[:peak_idx])
            except (TypeError, ValueError, AttributeError):
                logging.error("Doesn't find the longest linear segment")
                return (0, 0, 0, 0, 0, 0, 0)

            x1, y1, x2, y2 = loop[start][0], loop[start][1], loop[end - 1][0], loop[end - 1][1]
            ya = k * x_end + b

            y_peak = find_y_peak(kind, y_loop, k, b, x_loop)
            x_peak = loop[y_loop.index(y_peak)][0]

            y12 = (y_peak - (k * x_peak + b)) / 2

            p3 = closest_point(x_loop.index(x2), y_loop.index(y_peak), loop, (x1, y1 + y12), (x2, y2 + y12))

            e12 = p3[0] / del_x
            ep = x_peak / del_x
            yp = 2 * y12 / del_y
            draw(axe, x1, x2, x_end, y1, y2, ya, kind, x_peak, y_peak, k, b, y12, p3)
            if loop_num == 1:
                draw(plt, x1, x2, x_end, y1, y2, ya, kind, x_peak, y_peak, k, b, y12, p3)

            return (e12, ep, yp, k, b, y_peak, start, end)

        x_loop = [loop[z][0] for z in range(len(loop))]
        y_loop = [loop[z][1] for z in range(len(loop))]
        x_max_idx = x_loop.index(x_max)

        x_loop_a = x_loop[:x_max_idx + 1]
        y_loop_a = y_loop[:x_max_idx + 1]
        loopa = loop[:x_max_idx + 1]

        x_loop_c = x_loop[x_max_idx + 1:]
        y_loop_c = y_loop[x_max_idx + 1:]
        loopc = loop[x_max_idx + 1:]

        x_max_a = x_loop_a[-1]
        x_min_c = x_loop_c[-1] if x_loop_c else x_max

        e12a, epa, ypa, ka, ba, y_peak_a, start_a, end_a = counting(loopa, x_loop_a, y_loop_a, x_max_a, loop_num, "a")

        e12c, epc, ypc, kc, bc, y_peak_c, start_c, end_c = counting(loopc, x_loop_c, y_loop_c, x_min_c, loop_num, "c")

        p4 = closest_point(0, y_loop_c.index(y_peak_c), loopc, (x_loop[start_a], y_loop[start_a]), (x_loop[end_a], y_loop[end_a]))
        intersection_c_idx = loopc.index(p4)
        area_anodic_a = find_square(x_loop_a[end_a:], y_loop_a[end_a:], ka, ba)
        area_cathodic_a = find_square(x_loop_c[:intersection_c_idx + 1], y_loop_c[:intersection_c_idx + 1], ka, ba)
        spa = area_anodic_a + area_cathodic_a

        p5 = closest_point(0, y_loop_a.index(y_peak_a), loopa, (x_loop[start_c], y_loop[start_c]), (x_loop[end_c], y_loop[end_c]))
        intersection_a_idx = loopa.index(p5)
        area_cathodic_c = find_square(x_loop_c[end_c:], y_loop_c[end_c:], kc, bc)
        area_anodic_c = find_square(x_loop_a[:intersection_a_idx + 1], y_loop_a[:intersection_a_idx + 1], kc, bc)
        spc = area_cathodic_c + area_anodic_c

        return(e12a, epa, ypa, spa, e12c, epc, ypc, spc)

    for file_index, filepath in enumerate(mpr_files, start=1):
        df = read_mpr_file_safely(filepath)
        if df is None:
            failed_files.append(filepath.name)
            continue

        ewe = df.get("Ewe/V")
        i_main = df.get("<I>/mA")

        #Проверка наличия колонок в файле
        if ewe is None or i_main is None:
            logging.error(f"Missing required columns in {filepath.name}")
            failed_files.append(filepath.name)
            continue

        #Выделение точек окончания циклов
        endpoints=[]
        for a in range(1,len(ewe)-1):
            if ewe[a - 1] > ewe[a] < ewe[a + 1]:
                endpoints += [a]

        del_x = decimal.Decimal('1')
        del_y = decimal.Decimal('1')

        loops = []
        for z in range(len(endpoints)):
            start = 0 if z == 0 else endpoints[z - 1] + 1
            end = endpoints[z] + 1
            loop = [(decimal.Decimal(str(ewe[x] * float(del_x))), decimal.Decimal(str(i_main[x])) * del_y)
                    for x in range(start, end)]
            loops.append(loop)

        all_e12a, all_epa, all_ypa, all_sa, all_e12c, all_epc, all_ypc, all_sc = [], [], [], [], [], [], [], []
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

            e12a, epa, ypa, sa, e12c, epc, ypc, sc = Chart1(loop, del_x, del_y, axe, loop_index)
            all_e12a += [e12a]
            all_epa += [epa]
            all_ypa += [ypa]
            all_sa += [sa]
            all_e12c += [e12c]
            all_epc += [epc]
            all_ypc += [ypc]
            all_sc += [sc]
            print_log(loop_index, e12a, epa, ypa, sa, e12c, epc, ypc, sc)
        plt.show()

        data = {"E1/2a": all_e12a,
                "Epa": all_epa,
                "Ypa": all_ypa,
                "Spa": all_sa,
                "E1/2c": all_e12c,
                "Epc": all_epc,
                "Ypc": all_ypc,
                "Spc": all_sc,
                }
        d = pd.DataFrame(data)
        d.to_csv(INPUT_DIR/"data.csv", index=True, sep=";")


        # Add identification columns
        df.insert(0, 'source_file', filepath.name)
        df.insert(1, 'technique', extract_technique(filepath.name))
        df.insert(2, 'file_index', file_index)

        all_dataframes.append(df)
        successful_files.append((filepath.name, len(df)))


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
