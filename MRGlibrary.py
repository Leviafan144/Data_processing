#!/usr/bin/env python3
"""
Extract all data from Bio-Logic .mpr files and combine into single CSV.

Processes CV and PEIS files, handling different column structures.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from galvani import BioLogic

import math
import turtle as t
import decimal
from scipy.integrate import quad
decimal.getcontext().prec = 20

t.up()
t.tracer(0)
t.screensize(20000, 20000)

def perpendicularDistance(point, p1, p2):
    x0, y0 = point
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2:
        return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

    t = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if t < 0:
        x, y = x1, y1
    elif t > 1:
        x, y = x2, y2
    else:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

    return ((x0 - x) ** 2 + (y0 - y) ** 2) ** decimal.Decimal('0.5')


def DouglasPeucker(PointList, epsilon):
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(PointList)
    for i in range(1, end - 1):
        d = perpendicularDistance(PointList[i], PointList[0], PointList[end - 1])
        if d > dmax:
            index = i
            dmax = d

    ResultList = []

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        recResults1 = DouglasPeucker(PointList[0:index + 1], epsilon)
        recResults2 = DouglasPeucker(PointList[index:end], epsilon)

        # Build the result list
        ResultList = recResults1[0:len(recResults1) - 1] + recResults2[0:len(recResults2)]
    else:
        ResultList = [PointList[0], PointList[end - 1]]

    # Return the result
    return ResultList


# Configuration
INPUT_DIR = Path("C:\\pycharm\\PythonProject7")
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
        for z in range(i_start, i_finish):
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

    def functiona(loop, epsilon, del_x, del_y):
        x_loop = [loop[z][0] for z in range(len(loop))]
        y_loop = [loop[z][1] for z in range(len(loop))]
        y_max = max(y_loop)
        x_max = max(x_loop)
        loop_y_max = loop[y_loop.index(y_max)]

        loop_low = DouglasPeucker(loop[:loop.index(loop_y_max) // 2], epsilon)
        print(len(loop_low))

        for i in range(len(loop_low) - 2, 0, -1):
            x1, y1, x2, y2 = loop_low[i][0], loop_low[i][1], loop_low[i + 1][0], loop_low[i + 1][1]
            ya = (y2 - y1) * (x_max - x1) / (x2 - x1) + y1
            if loop[x_loop.index(x_max)][1] < ya < y_max * 2:
                break
        p1 = closest_point(loop.index(loop_y_max), loop.index(loop_x_max), loop, (x1, y1), (x2, y2))
        p2 = closest_point(x_loop.index(x2) + 10, loop.index(loop_y_max), loop, (x1, y1), (x2, y2))

        b = (y_max - ((y2 - y1) * (loop_y_max[0] - x1) / (x2 - x1) + y1)) / 2

        s = 0
        for z in range(loop.index(p2), loop.index(p1)):
            point1 = loop[z]
            point2 = loop[z + 1]
            ys = point2[1]
            xs = (point2[0] - point1[0])
            s += ys * xs
        x3, y3, x4, y4 = map(float, [p2[0], p2[1], p1[0], p1[1]])

        def integral(x):
            return (y4 - y3) * (x - x3) / (x4 - x3) + y3

        s_line = decimal.Decimal(str(quad(integral, x3, x4)[0]))
        s -= s_line
        s = s / del_x / del_y
        p3 = closest_point(x_loop.index(x2), loop.index(loop_y_max), loop, (x1, y1 + b), (x2, y2 + b))

        e12 = (p3[0] - dt * k) / del_x
        ep = (loop_y_max[0] - dt * k) / del_x
        yp = 2 * b / del_y

        t.goto(float(x1), float(y1))
        t.down()
        t.goto(float(x_max), float(ya))
        t.up()
        t.goto(float(x1), float(y1))
        t.dot(10, "blue")
        t.goto(float(x2), float(y2))
        t.dot(10, "blue")
        t.goto(float(loop_y_max[0]), float(loop_y_max[1]))
        t.down()
        t.goto(float(loop_y_max[0]), 0)
        t.up()
        t.goto(float(p2[0]), float(p2[1]))
        t.dot(7, "blue")
        t.goto(float(p1[0]), float(p1[1]))
        t.dot(7, "blue")
        t.goto(float(loop_y_max[0]), float(y_max - b))
        t.dot(7, "blue")
        t.goto(float(x2), float(y2 + b))
        t.down()
        t.goto(float(x_max), float(ya + b))
        t.up()
        t.goto(float(p3[0]), float(p3[1]))
        t.dot(7, "blue")

        for point in loop_low:
            t.goto(float(point[0]), float(point[1]))
            t.dot(5, "green")

        return (e12, ep, yp, s)

    def functionc(loop, epsilon, del_x, del_y):
        x_loop = [loop[z][0] for z in range(len(loop))]
        y_loop = [loop[z][1] for z in range(len(loop))]
        y_min = min(y_loop)
        x_min = min(x_loop)
        loop_y_min = loop[y_loop.index(y_min)]
        loop_x_min = loop[x_loop.index(x_min)]

        loop_low = DouglasPeucker(loop[:loop.index(loop_y_min) // 2], epsilon)
        print(len(loop_low))

        for i in range(len(loop_low) - 2, 0, -1):
            x1, y1, x2, y2 = loop_low[i][0], loop_low[i][1], loop_low[i + 1][0], loop_low[i + 1][1]
            ya = (y2 - y1) * (x_min - x1) / (x2 - x1) + y1
            if loop[x_loop.index(x_min)][1] > ya > y_min * 2:
                break
        p1 = closest_point(loop.index(loop_y_min), loop.index(loop_x_min), loop, (x1, y1), (x2, y2))
        p2 = closest_point(x_loop.index(x2) + 10, loop.index(loop_y_min), loop, (x1, y1), (x2, y2))

        b = (y_min - ((y2 - y1) * (loop_y_min[0] - x1) / (x2 - x1) + y1)) / 2

        s = 0
        for z in range(loop.index(p2), loop.index(p1)):
            point1 = loop[z]
            point2 = loop[z + 1]
            ys = point2[1]
            xs = (point2[0] - point1[0])
            s += ys * xs
        x3, y3, x4, y4 = map(float, [p2[0], p2[1], p1[0], p1[1]])

        def integral(x):
            return (y3 - y4) * (x - x3) / (x4 - x3) + y3

        s_line = decimal.Decimal(str(quad(integral, x3, x4)[0]))
        s -= s_line
        s = s / del_x / del_y
        p3 = closest_point(x_loop.index(x2), loop.index(loop_y_min), loop, (x1, y1 + b), (x2, y2 + b))

        e12 = (p3[0] - dt * k) / del_x
        ep = (loop_y_min[0] - dt * k) / del_x
        yp = 2 * b / del_y

        t.goto(float(x1), float(y1))
        t.down()
        t.goto(float(x_min), float(ya))
        t.up()
        t.goto(float(x1), float(y1))
        t.dot(10, "blue")
        t.goto(float(x2), float(y2))
        t.dot(10, "blue")
        t.goto(float(loop_y_min[0]), float(loop_y_min[1]))
        t.down()
        t.goto(float(loop_y_min[0]), 0)
        t.up()
        t.goto(float(p2[0]), float(p2[1]))
        t.dot(7, "blue")
        t.goto(float(p1[0]), float(p1[1]))
        t.dot(7, "blue")
        t.goto(float(loop_y_min[0]), float(y_min - b))
        t.dot(7, "blue")
        t.goto(float(x2), float(y2 + b))
        t.down()
        t.goto(float(x_min), float(ya + b))
        t.up()
        t.goto(float(p3[0]), float(p3[1]))
        t.dot(7, "blue")

        for point in loop_low:
            t.goto(float(point[0]), float(point[1]))
            t.dot(5, "green")

        return (e12, ep, yp, s)

    for file_index, filepath in enumerate(mpr_files, start=1):
        df = read_mpr_file_safely(filepath)

        ewe = df.get("Ewe/V")
        i = df.get("<I>/mA")

        #Выделение точек окончания циклов
        w=[]
        for a in range(1,len(ewe)-1):
            if ewe[a - 1] > ewe[a] < ewe[a + 1]:
                w += [a]

        dt = decimal.Decimal('700')
        del_x = 1000
        del_y = 150
        loops = [[(decimal.Decimal(str(ewe[x] * del_x)) +dt*z, decimal.Decimal(str(i[x]*del_y))) for x in range(0+482*z, 483+482*z)] for z in range(0,len(w))]

        all_e12a, all_epa, all_ypa, all_sa, all_e12c, all_epc, all_ypc, all_sc = [], [], [], [], [], [], [], []
        for k in range(len(loops)):
            loop = loops[k]
            x_loop = [loop[z][0] for z in range(len(loop))]
            x_max = max(x_loop)
            loop_x_max = loop[x_loop.index(x_max)]
            loopa = loop[:x_loop.index(x_max)+1]
            e12a, epa, ypa, sa = functiona(loopa, 0.5, del_x, del_y)
            loopc = loop[x_loop.index(x_max) + 1:]
            e12c, epc, ypc, sc = functionc(loopc, 0.5, del_x, del_y)
            all_e12a += [e12a]
            all_epa += [epa]
            all_ypa += [ypa]
            all_sa += [sa]
            all_e12c += [e12c]
            all_epc += [epc]
            all_ypc += [ypc]
            all_sc += [sc]
            print_log(k, e12a, epa, ypa, sa, e12c, epc, ypc, sc)

            for point in loop:
                t.goto(float(point[0]), float(point[1]))
                t.dot(3,"red")

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
        d.to_csv("data.csv", index=True, sep=";")

        t.goto(0,0)
        t.down()
        t.goto(100000, 0)
        t.up()
        t.done()


        if df is not None:
            # Add identification columns
            df.insert(0, 'source_file', filepath.name)
            df.insert(1, 'technique', extract_technique(filepath.name))
            df.insert(2, 'file_index', file_index)

            all_dataframes.append(df)
            successful_files.append((filepath.name, len(df)))
        else:
            failed_files.append(filepath.name)

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
