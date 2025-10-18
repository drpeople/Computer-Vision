# #### Localization
"""
Create *pivoted* summary CSVs:

    row    = the exact folder name that ends with 'Localization'
    column = invariance parameter (Scale-factor, Sigma, Noise-Std, Rotation-angle)
    value  = Mean IoU

The script walks through every sub-folder of BASE_DIR that ends with
'Localization', looks for the four *xxx_results.csv* files, concatenates them
type-by-type, pivots, and writes the four 'aggregated_*.csv' files.
"""

import os
import glob
import pandas as pd   # pip install pandas

# ----------------------------------------------------------------------
# 1️⃣  EDIT THIS IF YOUR ROOT FOLDER MOVES
# ----------------------------------------------------------------------
BASE_DIR = r"C:\Users\goker\Desktop\results"

# ----------------------------------------------------------------------
# 2️⃣  4 experiment types we’re looking for
# ----------------------------------------------------------------------
EXPERIMENTS = {
    "noise_results":   {"param": "Noise Std",
                        "outfile": "aggregated_noise_results.csv"},
    "rotation_results":{"param": "Rotation Angle",
                        "outfile": "aggregated_rotation_results.csv"},
    "scale_results":   {"param": "Scale Factor",
                        "outfile": "aggregated_scale_results.csv"},
    "blur_results":    {"param": "Sigma",
                        "outfile": "aggregated_blur_results.csv"},
}

# ----------------------------------------------------------------------
# 3️⃣  COLLECT ALL CSVs
# ----------------------------------------------------------------------
buckets = {k: [] for k in EXPERIMENTS}           # lists of DataFrames

for root, dirs, _ in os.walk(BASE_DIR):
    for d in dirs:
        if not d.endswith("Localization"):
            continue

        loc_path = os.path.join(root, d)         # full path to that model-folder
        for key in EXPERIMENTS:
            pattern = os.path.join(loc_path, f"*{key}*.csv")
            for csv_file in glob.glob(pattern):
                try:
                    df = pd.read_csv(csv_file)
                except Exception as err:
                    print(f"[WARN]  could not read {csv_file!r}: {err}")
                    continue

                df["LocalizationFolder"] = d     # keep the *whole* folder name
                buckets[key].append(df)

# ----------------------------------------------------------------------
# 4️⃣  PIVOT & WRITE
# ----------------------------------------------------------------------
for key, frames in buckets.items():
    if not frames:
        print(f"[INFO]  no '{key}' files found – skipping.")
        continue

    info = EXPERIMENTS[key]
    param_col = info["param"]

    big = pd.concat(frames, ignore_index=True)

    # rows = folder, columns = parameter value, cell = Mean IoU
    mat = (big
           .pivot_table(index="LocalizationFolder",
                        columns=param_col,
                        values="Mean IoU",
                        aggfunc="first"))

    # put columns in natural numeric order if possible
    try:
        mat = mat[mat.columns.astype(float).sort_values()]
    except ValueError:          # columns are not numeric – just sort as strings
        mat = mat[sorted(mat.columns)]

    out_path = os.path.join(BASE_DIR, info["outfile"])
    mat.reset_index().to_csv(out_path, index=False)
    print(f"[OK]   wrote {out_path}   ({mat.shape[0]} models × {mat.shape[1]-1} settings)")


#Recognition
# """
# Aggregate Recognition-experiment CSVs and create four files
#
#     aggregated_noise_results_recognition.csv
#     aggregated_rotation_results_recognition.csv
#     aggregated_scale_results_recognition.csv
#     aggregated_blur_results_recognition.csv
#
# Each file has
#
#     row    = folder name that ends with 'Recognition'
#     column = invariance value (noise_std • angle • scale_factor • blur_radius)
#     value  = Accuracy
#
# The script is tolerant of column-name variations such as
# 'Accuracy', 'accuracy ', 'accuracy_score', 'Acc', etc.
# """
#
# import os
# import re
# import glob
# import pandas as pd      # pip install pandas
#
# # ───────────────────────────────────────────────────────────────
# # 1.  ROOT FOLDER
# # ───────────────────────────────────────────────────────────────
# BASE_DIR = r"C:\Users\goker\Desktop\results"   # ← change if needed
#
# # ───────────────────────────────────────────────────────────────
# # 2.  EXPERIMENT DEFINITIONS (your updated mapping)
# # ───────────────────────────────────────────────────────────────
# EXPERIMENTS = {
#     "noise_results": {
#         "param":   "noise_std",
#         "outfile": "aggregated_noise_results_recognition.csv",
#     },
#     "rotation_results": {
#         "param":   "angle",
#         "outfile": "aggregated_rotation_results_recognition.csv",
#     },
#     "scale_results": {
#         "param":   "scale_factor",
#         "outfile": "aggregated_scale_results_recognition.csv",
#     },
#     "blur_results": {
#         "param":   "blur_radius",
#         "outfile": "aggregated_blur_results_recognition.csv",
#     },
# }
#
# # ───────────────────────────────────────────────────────────────
# # 3.  FLEXIBLE COLUMN DETECTORS
# # ───────────────────────────────────────────────────────────────
# def normalise(name: str) -> str:
#     """lower-case and drop spaces / underscores for easy matching"""
#     return re.sub(r"[\s_]", "", name).lower()
#
#
# def find_column(columns, target_norm):
#     """Return the actual column name whose normalised form matches target_norm."""
#     for col in columns:
#         if normalise(col) == target_norm:
#             return col
#     return None
#
#
# # normalised tokens we care about
# ACC_NORM = normalise("accuracy")
#
# PARAM_NORM = {k: normalise(v["param"]) for k, v in EXPERIMENTS.items()}
#
# # ───────────────────────────────────────────────────────────────
# # 4.  COLLECT CSVs
# # ───────────────────────────────────────────────────────────────
# buckets = {key: [] for key in EXPERIMENTS}          # experiment → list[DataFrame]
#
# for root, dirs, _ in os.walk(BASE_DIR):
#     for folder in dirs:
#         if not folder.endswith("Recognition"):
#             continue
#
#         rec_path = os.path.join(root, folder)
#
#         for key, cfg in EXPERIMENTS.items():
#             pattern = os.path.join(rec_path, f"*{key}*.csv")
#             for csv_path in glob.glob(pattern):
#                 try:
#                     df = pd.read_csv(csv_path)
#                 except Exception as err:
#                     print(f"[WARN] cannot read {csv_path!r}: {err}")
#                     continue
#
#                 df.columns = df.columns.str.strip()          # trim spaces
#
#                 # locate Accuracy column
#                 acc_col = find_column(df.columns, ACC_NORM)
#                 if acc_col is None:
#                     print(f"[WARN] '{csv_path}' skipped – no Accuracy column.")
#                     continue
#
#                 # locate parameter column
#                 param_norm = PARAM_NORM[key]
#                 param_col = find_column(df.columns, param_norm)
#                 if param_col is None:
#                     print(f"[WARN] '{csv_path}' skipped – "
#                           f"no '{cfg['param']}' column.")
#                     continue
#
#                 # keep only what we need + folder name
#                 df = df[[acc_col, param_col]].copy()
#                 df.rename(columns={acc_col: "Accuracy",
#                                    param_col: cfg["param"]}, inplace=True)
#                 df["RecognitionFolder"] = folder
#                 buckets[key].append(df)
#
# # ───────────────────────────────────────────────────────────────
# # 5.  PIVOT & SAVE
# # ───────────────────────────────────────────────────────────────
# for key, dfs in buckets.items():
#     if not dfs:
#         print(f"[INFO] no usable '{key}' CSVs found – nothing written.")
#         continue
#
#     cfg = EXPERIMENTS[key]
#     param_col = cfg["param"]
#
#     big = pd.concat(dfs, ignore_index=True)
#
#     matrix = (big
#               .pivot_table(index="RecognitionFolder",
#                            columns=param_col,
#                            values="Accuracy",
#                            aggfunc="first"))
#
#     # order columns numerically if they are numbers
#     try:
#         matrix = matrix[matrix.columns.astype(float).sort_values()]
#     except ValueError:
#         matrix = matrix[matrix.columns.sort_values()]
#
#     out_path = os.path.join(BASE_DIR, cfg["outfile"])
#     matrix.reset_index().to_csv(out_path, index=False)
#     print(f"[OK] wrote {out_path} "
#           f"({matrix.shape[0]} folders × {matrix.shape[1]-1} settings)")

# # Segmentation
# """
# Aggregate Segmentation experiment CSVs into four 'aggregated_*_segmentation.csv'
# matrix files.
#
# * rows   = folder name that ends with 'Segmentation'
# * cols   = invariance parameter (Noise Std, Rotation Angle, Scale Factor, Sigma)
# * values = Mean IoU (column name can be 'Mean IoU', 'MeanIoU', 'mean_iou', …)
#
# If a CSV lacks any recognisable Mean-IoU column it is skipped with a warning.
# """
#
# import os
# import glob
# import re
# import pandas as pd      # pip install pandas
#
# # ───────────────────────────────────────────────────────────────
# # 1.  CONFIG
# # ───────────────────────────────────────────────────────────────
# BASE_DIR = r"C:\Users\goker\Desktop\results"   # <-- change if needed
#
# EXPERIMENTS = {
#     "noise_results": {
#         "param":   "noise_std",
#         "outfile": "aggregated_noise_results_segmentation.csv",
#     },
#     "rotation_results": {
#         "param":   "rotation_angle",
#         "outfile": "aggregated_rotation_results_segmentation.csv",
#     },
#     "scale_results": {
#         "param":   "scale_factor",
#         "outfile": "aggregated_scale_results_segmentation.csv",
#     },
#     "blur_results": {
#         "param":   "sigma",
#         "outfile": "aggregated_blur_results_segmentation.csv",
#     },
# }
#
# def find_mean_iou_column(columns):
#     """
#     Return the column name that represents Mean IoU, or None if not found.
#     Comparison is case-insensitive and ignores spaces and underscores.
#     """
#     for col in columns:
#         # remove spaces and underscores, lowercase → e.g. "Mean_IoU " -> "meaniou"
#         norm = re.sub(r"[\s_]+", "", col).lower()
#         if norm == "meaniou":
#             return col
#     return None
#
# # ───────────────────────────────────────────────────────────────
# # 2.  GATHER FILES
# # ───────────────────────────────────────────────────────────────
# buckets = {key: [] for key in EXPERIMENTS}  # exp-type → list[DataFrame]
#
# for root, dirs, _ in os.walk(BASE_DIR):
#     for d in dirs:
#         if not d.endswith("Segmentation"):
#             continue
#
#         seg_path = os.path.join(root, d)
#         for key, cfg in EXPERIMENTS.items():
#             pattern = os.path.join(seg_path, f"*{key}*.csv")
#             for csv_path in glob.glob(pattern):
#                 try:
#                     df = pd.read_csv(csv_path)
#                 except Exception as err:
#                     print(f"[WARN] can't read {csv_path!r}: {err}")
#                     continue
#
#                 # strip whitespace from headers to avoid ' Mean IoU'
#                 df.columns = df.columns.str.strip()
#
#                 mean_col = find_mean_iou_column(df.columns)
#                 if mean_col is None:
#                     print(f"[WARN] '{csv_path}' skipped – can't find Mean IoU column.")
#                     continue
#
#                 needed = [mean_col, cfg["param"]]
#                 if any(c not in df.columns for c in needed):
#                     missing = [c for c in needed if c not in df.columns]
#                     print(f"[WARN] '{csv_path}' missing {missing}; skipped.")
#                     continue
#
#                 df = df[needed].copy()
#                 df.rename(columns={mean_col: "mean_iou"}, inplace=True)
#                 df["SegmentationFolder"] = d
#                 buckets[key].append(df)
#
# # ───────────────────────────────────────────────────────────────
# # 3.  PIVOT & WRITE
# # ───────────────────────────────────────────────────────────────
# for key, dfs in buckets.items():
#     if not dfs:
#         print(f"[INFO] no usable '{key}' CSVs found – nothing written.")
#         continue
#
#     param_col = EXPERIMENTS[key]["param"]
#     big = pd.concat(dfs, ignore_index=True)
#
#     matrix = big.pivot_table(
#         index="SegmentationFolder",
#         columns=param_col,
#         values="mean_iou",
#         aggfunc="first"
#     )
#
#     # order columns numerically if possible
#     try:
#         matrix = matrix[matrix.columns.astype(float).sort_values()]
#     except ValueError:
#         matrix = matrix[matrix.columns.sort_values()]
#
#     out_path = os.path.join(BASE_DIR, EXPERIMENTS[key]["outfile"])
#     matrix.reset_index().to_csv(out_path, index=False)
#     print(
#         f"[OK] wrote {out_path}   "
#         f"({matrix.shape[0]} folders × {matrix.shape[1]} settings)"
#     )
