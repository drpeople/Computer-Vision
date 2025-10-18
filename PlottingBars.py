import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")


def plot_localization_invariances(root_dir):
    invariance_types = ["blur", "noise", "rotation", "scale"]
    x_axis_map = {
        "blur": "Sigma",
        "rotation": "Rotation Angle",
        "scale": "Scale Factor"
    }
    plot_data = {inv: {} for inv in invariance_types}

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path) and "Localization" in folder_name:
            model_name = folder_name.replace(" Localization", "").strip()
            print(f"Processing folder for model: {model_name}")

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".csv"):
                    for inv_type in invariance_types:
                        if inv_type in file_name.lower():
                            csv_path = os.path.join(folder_path, file_name)
                            try:
                                df = pd.read_csv(csv_path)
                                print(f"Loaded {csv_path} with {len(df)} rows; columns: {df.columns.tolist()}")
                                plot_data[inv_type][model_name] = df
                            except Exception as e:
                                print(f"Error reading {csv_path}: {e}")

    for inv_type in invariance_types:
        if not plot_data[inv_type]:
            print(f"No data found for {inv_type}.")
            continue

        plt.figure(figsize=(14, 6))

        if inv_type == "noise":
            possible_noise_columns = ["Noise Std", "Noise Level"]
        else:
            expected_x = x_axis_map[inv_type]

        plotted_any = False

        if inv_type == "rotation":
            # Setup for stacked bar plot
            all_angles = sorted(set(
                angle for df in plot_data[inv_type].values() if "Rotation Angle" in df.columns
                for angle in df["Rotation Angle"].unique()
            ))

            model_names = sorted(plot_data[inv_type].keys())
            n_angles = len(all_angles)
            x = np.arange(n_angles)
            bar_width = 0.6

            # Preload data per model
            angle_to_index = {angle: idx for idx, angle in enumerate(all_angles)}
            base = np.zeros(n_angles)
            colors = sns.color_palette("tab20", n_colors=len(model_names))

            for i, model_name in enumerate(model_names):
                df = plot_data[inv_type][model_name]
                if "Mean IoU" not in df.columns or "Rotation Angle" not in df.columns:
                    continue

                y_values = np.zeros(n_angles)
                for _, row in df.iterrows():
                    angle = row["Rotation Angle"]
                    if angle in angle_to_index:
                        idx = angle_to_index[angle]
                        y_values[idx] = row["Mean IoU"]

                bars = plt.bar(x, y_values, bottom=base, width=bar_width, label=model_name, color=colors[i], edgecolor='white')

                # Add value labels inside the bars if the segment is tall enough
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.5:
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + height / 2,
                            f"{height:.2f}",
                            ha='center', va='center', fontsize=8, color='white', weight='bold'
                        )

                base += y_values
                plotted_any = True

            x_labels = [f"{int(a)}°" for a in all_angles]
            plt.xticks(ticks=x, labels=x_labels, fontsize=10)
            x_col = "Rotation Angle"

        else:
            for model_name, df in plot_data[inv_type].items():
                if "Mean IoU" not in df.columns:
                    continue

                if inv_type == "noise":
                    x_col = next((col for col in ["Noise Std", "Noise Level"] if col in df.columns), None)
                    if not x_col:
                        continue
                else:
                    x_col = expected_x
                    if x_col not in df.columns:
                        continue

                x_values = df[x_col]
                y_values = df["Mean IoU"]
                plt.plot(x_values, y_values, marker='o', label=model_name)
                plotted_any = True

        if not plotted_any:
            print(f"No valid data plotted for {inv_type}.")
            plt.close()
            continue

        plt.title(f"{inv_type.capitalize()} Invariance: Mean IoU vs. {x_col}", fontsize=14, weight='bold')
        plt.xlabel(x_col, fontsize=12, weight='bold')
        plt.ylabel("Mean IoU", fontsize=12, weight='bold')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    results_path = r"C:\Users\goker\Desktop\results"
    plot_localization_invariances(results_path)

##### RECOGNITION
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def plot_recognition_invariances(root_dir):
#     """
#     Scans the 'root_dir' for subfolders containing 'Recognition' in their name,
#     finds CSV files for each of the four invariance types (blur, noise, rotation, scale),
#     and plots them together on one figure per invariance type.
#
#     For Recognition CSV files:
#       - Y-axis: 'accuracy'
#       - X-axis depends on the invariance type:
#           * blur: 'blur_radius'
#           * noise: 'noise_std' (fallback to 'noise_level' if needed)
#           * rotation: 'angle'
#           * scale: 'scale_factor'
#     """
#     invariance_types = ["blur", "noise", "rotation", "scale"]
#
#     # Expected X-axis columns (noise handled separately)
#     x_axis_map = {
#         "blur": "blur_radius",
#         "rotation": "angle",
#         "scale": "scale_factor"
#     }
#
#     # Container to store data: {invariance_type: {model_name: DataFrame}}
#     plot_data = {inv: {} for inv in invariance_types}
#
#     # 1) Loop through subfolders and select those with "Recognition"
#     for folder_name in os.listdir(root_dir):
#         folder_path = os.path.join(root_dir, folder_name)
#         if os.path.isdir(folder_path) and "Recognition" in folder_name:
#             model_name = folder_name.replace(" Recognition", "").strip()
#             print(f"Processing folder for model: {model_name}")
#
#             # Look for CSV files for each invariance type
#             for file_name in os.listdir(folder_path):
#                 if file_name.endswith(".csv"):
#                     file_lower = file_name.lower()
#                     for inv_type in invariance_types:
#                         if inv_type in file_lower:
#                             csv_path = os.path.join(folder_path, file_name)
#                             try:
#                                 df = pd.read_csv(csv_path)
#                                 print(f"Loaded {csv_path} with {len(df)} rows; columns: {df.columns.tolist()}")
#                                 plot_data[inv_type][model_name] = df
#                             except Exception as e:
#                                 print(f"Error reading {csv_path}: {e}")
#
#     # 2) Plot each invariance type in its own figure
#     for inv_type in invariance_types:
#         if not plot_data[inv_type]:
#             print(f"No data found for {inv_type} in Recognition folders.")
#             continue
#
#         # Make a bigger figure to accommodate the legend on the right
#         plt.figure(figsize=(10, 6))
#
#         # Determine the expected X-axis column
#         if inv_type == "noise":
#             expected_x = "noise_std"
#         else:
#             expected_x = x_axis_map[inv_type]
#
#         plotted_any = False
#
#         for model_name, df in plot_data[inv_type].items():
#             # For Recognition, expected Y-axis column is 'accuracy'
#             if "accuracy" not in df.columns:
#                 print(
#                     f"CSV for {model_name} ({inv_type}) is missing the 'accuracy' column. Found: {df.columns.tolist()}")
#                 continue
#
#             # Handle noise CSVs that might have 'noise_level' instead of 'noise_std'
#             if inv_type == "noise" and expected_x not in df.columns:
#                 if "noise_level" in df.columns:
#                     expected_x = "noise_level"
#                 else:
#                     print(
#                         f"CSV for {model_name} (noise) is missing '{expected_x}' or 'noise_level'. Found: {df.columns.tolist()}")
#                     continue
#             elif inv_type != "noise" and expected_x not in df.columns:
#                 print(
#                     f"CSV for {model_name} ({inv_type}) is missing expected X-axis column '{expected_x}'. Found: {df.columns.tolist()}")
#                 continue
#
#             x_values = df[expected_x]
#             y_values = df["accuracy"]
#
#             plt.plot(x_values, y_values, marker='o', label=model_name)
#             plotted_any = True
#
#         if not plotted_any:
#             print(f"No valid data plotted for {inv_type}.")
#             plt.close()
#             continue
#
#         plt.title(f"{inv_type.capitalize()} Invariance: Accuracy vs. {expected_x}")
#         plt.xlabel(expected_x)
#         plt.ylabel("accuracy")
#
#         # Place legend outside the plot on the right
#         plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#
#         # Adjust the subplot so that the legend doesn't squeeze the plot
#         plt.subplots_adjust(right=0.75)  # Increase space on the right side
#
#         plt.grid(True)
#
#         plt.show()
#         plt.close()
#
#
# if __name__ == "__main__":
#     # Change this to the path of your "results" directory
#     results_path = r"C:\Users\goker\Desktop\results"
#     plot_recognition_invariances(results_path)

### Segmentation
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def plot_segmentation_invariances(root_dir):
#     """
#     Scans the 'root_dir' for subfolders containing 'Segmentation' in their name,
#     reads CSV files for each invariance type (blur, noise, rotation, scale) from each model,
#     and plots them together on one figure per invariance type.
#
#     For Segmentation CSV files, the expected columns are:
#       - Y-axis: 'mean_iou'
#       - X-axis:
#           * blur: 'sigma'
#           * noise: 'noise_std' (fallback to 'noise_level' if needed)
#           * rotation: 'rotation_angle'
#           * scale: 'scale_factor'
#     """
#     # Define the invariance types
#     invariance_types = ["blur", "noise", "rotation", "scale"]
#
#     # Mapping for expected X-axis columns for Segmentation (noise handled separately)
#     x_axis_map = {
#         "blur": "sigma",
#         "rotation": "rotation_angle",
#         "scale": "scale_factor"
#     }
#
#     # Dictionary to store the data: {invariance_type: {model_name: DataFrame}}
#     plot_data = {inv: {} for inv in invariance_types}
#
#     # Traverse subfolders looking for "Segmentation" in their name
#     for folder_name in os.listdir(root_dir):
#         folder_path = os.path.join(root_dir, folder_name)
#         if os.path.isdir(folder_path) and "Segmentation" in folder_name:
#             # Derive model name by removing " Segmentation"
#             model_name = folder_name.replace(" Segmentation", "").strip()
#             print(f"Processing folder for model: {model_name}")
#
#             # Look for CSV files that match each invariance type
#             for file_name in os.listdir(folder_path):
#                 if file_name.endswith(".csv"):
#                     file_lower = file_name.lower()
#                     for inv_type in invariance_types:
#                         if inv_type in file_lower:
#                             csv_path = os.path.join(folder_path, file_name)
#                             try:
#                                 df = pd.read_csv(csv_path)
#                                 print(f"Loaded {csv_path} with {len(df)} rows; columns: {df.columns.tolist()}")
#                                 plot_data[inv_type][model_name] = df
#                             except Exception as e:
#                                 print(f"Error reading {csv_path}: {e}")
#
#     # Plot each invariance type on its own figure
#     for inv_type in invariance_types:
#         if not plot_data[inv_type]:
#             print(f"No data found for {inv_type} in Segmentation folders.")
#             continue
#
#         # Increase figure size and reserve space for the legend
#         plt.figure(figsize=(10, 6))
#
#         # Determine expected X-axis column based on invariance type
#         if inv_type == "noise":
#             expected_x = "noise_std"
#         else:
#             expected_x = x_axis_map[inv_type]
#
#         plotted_any = False
#
#         for model_name, df in plot_data[inv_type].items():
#             # Check for the required Y-axis column ('mean_iou')
#             if "mean_iou" not in df.columns:
#                 print(f"CSV for {model_name} ({inv_type}) is missing 'mean_iou'. Found: {df.columns.tolist()}")
#                 continue
#
#             # For noise invariance, if 'noise_std' is not found, check for 'noise_level'
#             if inv_type == "noise" and expected_x not in df.columns:
#                 if "noise_level" in df.columns:
#                     expected_x = "noise_level"
#                 else:
#                     print(
#                         f"CSV for {model_name} (noise) is missing '{expected_x}' or 'noise_level'. Found: {df.columns.tolist()}")
#                     continue
#             elif inv_type != "noise" and expected_x not in df.columns:
#                 print(
#                     f"CSV for {model_name} ({inv_type}) is missing expected X-axis column '{expected_x}'. Found: {df.columns.tolist()}")
#                 continue
#
#             x_values = df[expected_x]
#             y_values = df["mean_iou"]
#
#             plt.plot(x_values, y_values, marker='o', label=model_name)
#             plotted_any = True
#
#         if not plotted_any:
#             print(f"No valid data plotted for {inv_type}.")
#             plt.close()
#             continue
#
#         plt.title(f"{inv_type.capitalize()} Invariance: mean_iou vs. {expected_x}")
#         plt.xlabel(expected_x)
#         plt.ylabel("mean_iou")
#
#         # Place legend outside the plot on the right
#         plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#
#         # Adjust subplot to ensure the plot is not squeezed by the legend
#         plt.subplots_adjust(right=0.75)
#         plt.grid(True)
#
#         plt.show()
#         plt.close()
#
#
# if __name__ == "__main__":
#     # Change this to the path of your "results" directory
#     results_path = r"C:\Users\goker\Desktop\results"
#     plot_segmentation_invariances(results_path)
