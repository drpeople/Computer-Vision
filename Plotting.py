### Localization
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_localization_invariances(root_dir):
    """
    Scans the 'root_dir' for subfolders containing 'Localization' in their name,
    finds CSV files for each of the four invariance types, and plots them together.

    - X-axis depends on invariance type:
        * blur -> 'Sigma'
        * noise -> 'Noise Std' OR 'Noise Level'
        * rotation -> 'Rotation Angle'
        * scale -> 'Scale Factor'
    - Y-axis is 'Mean IoU'
    """
    # Define the invariance types we care about
    invariance_types = ["blur", "noise", "rotation", "scale"]

    # Map invariance type to the expected column name for X-axis (for noise, handled separately)
    x_axis_map = {
        "blur": "Sigma",
        "rotation": "Rotation Angle",
        "scale": "Scale Factor"
    }

    # This will hold DataFrames by [invariance_type][model_name]
    plot_data = {inv: {} for inv in invariance_types}

    # 1) Collect data from each "Localization" folder
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # Check if folder name contains 'Localization'
        if os.path.isdir(folder_path) and "Localization" in folder_name:
            # Derive model name (adjust if your naming is different)
            model_name = folder_name.replace(" Localization", "").strip()
            print(f"Processing folder for model: {model_name}")

            # Look for CSV files that contain our invariance keywords
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

    # 2) Plot each invariance type on its own figure
    for inv_type in invariance_types:
        if not plot_data[inv_type]:
            print(f"No data found for {inv_type}.")
            continue

        # Increase figure size and reserve space for the legend
        plt.figure(figsize=(10, 6))

        # Determine the X-axis column
        if inv_type == "noise":
            possible_noise_columns = ["Noise Std", "Noise Level"]
        else:
            expected_x = x_axis_map[inv_type]

        plotted_any = False

        for model_name, df in plot_data[inv_type].items():
            # Check for the required Y-axis column ('Mean IoU')
            if "Mean IoU" not in df.columns:
                print(f"CSV for {model_name} ({inv_type}) is missing 'Mean IoU'. Found: {df.columns.tolist()}")
                continue

            # Figure out the X-axis column for the current invariance type
            if inv_type == "noise":
                x_col = None
                for candidate in possible_noise_columns:
                    if candidate in df.columns:
                        x_col = candidate
                        break
                if not x_col:
                    print(f"CSV for {model_name} (noise) does not have a recognized noise column. Found: {df.columns.tolist()}")
                    continue
            else:
                x_col = expected_x
                if x_col not in df.columns:
                    print(f"CSV for {model_name} ({inv_type}) is missing expected X-axis column '{x_col}'. Found: {df.columns.tolist()}")
                    continue

            x_values = df[x_col]
            y_values = df["Mean IoU"]

            plt.plot(x_values, y_values, marker='o', label=model_name)
            plotted_any = True

        if not plotted_any:
            print(f"No valid data plotted for {inv_type}.")
            plt.close()
            continue

        plt.title(f"{inv_type.capitalize()} Invariance: Mean IoU vs. {x_col}")
        plt.xlabel(x_col)
        plt.ylabel("Mean IoU")

        # Place legend outside the plot on the right
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(right=0.75)
        plt.grid(True)

        plt.show()
        plt.close()


if __name__ == "__main__":
    # Update this path to your 'results' directory
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
