import ast
import os
import subprocess
import numpy as np
import scipy
import cv2
import openpyxl
import pandas as pd
# Path to the YOLO script
yolo_script_path = "yolo_opencv.py"

# YOLO configuration
config_file = "yolov3.cfg"
weights_file = "yolov3.weights"
classes_file = "yolov3.txt"

# Directory containing the images
images_directory = r"F:\object-detection-opencv-master\Frames"
output_directory = r"F:\object-detection-opencv-master"

def count_files_in_folder(folder_path):
    # Use os.listdir to get a list of all files and directories in the specified folder
    all_files = os.listdir(folder_path)

    # Use a list comprehension to filter out only files (not directories)
    files_only = [file for file in all_files if os.path.isfile(os.path.join(folder_path, file))]

    # Return the count of files
    return len(files_only)

# Specify the path to the folder you want to count files in
folder_path = r"F:\object-detection-opencv-master\Frames"
image_files = [f for f in os.listdir(images_directory) if f.endswith(".jpg")]
frame_bounding_boxes = {}
# Call the function and print the result
file_count = count_files_in_folder(folder_path)
#print(f"Number of files in the folder: {file_count}")
image_data_list = []
for i in range(file_count):
    if i % 9 == 1:
        # Retrieve the corresponding image file
        image_file = image_files[i // 9] if i < len(image_files) * 9 else None

        if image_file:
            # Construct the full path to the image
            image_path = os.path.join(images_directory, image_file)

            # Construct the command to run YOLO on the current image
            command = f"python {yolo_script_path} --image {image_path} --config {config_file} --weights {weights_file} --classes {classes_file}"
            # Run the YOLO script using subprocess
            subprocess.run(command, shell=True)
        else:
            print('Not Iframe')
    
#mat_file_path = 'output_all.mat'
#scipy.io.savemat(mat_file_path, {'bounding_boxes': list(frame_bounding_boxes.values())})
#print("All Bounding Boxes:")
#print(image_data_list)

'''New part'''
mat_file_path = 'output_all.mat'
excel_file_path = 'df.xlsx'
if os.path.exists(mat_file_path):
    existing_data = scipy.io.loadmat(mat_file_path)
scipy.io.savemat(mat_file_path, {'bounding_boxes': list(frame_bounding_boxes.values())})

if os.path.exists(excel_file_path):
    df_existing = pd.read_excel(excel_file_path)
df = pd.DataFrame(image_data_list)
df_combined = df_existing.append(df, ignore_index=True)
df_combined.to_excel(excel_file_path, index=False)





#df = pd.DataFrame(image_data_list)
#df2 = pd.DataFrame([pd.Series(x) for x in df.bounding_boxes])
#df2.columns = ['bounding_box_{}'.format(x+1) for x in df2.columns]
#df2["image_id"] = df["image_id"].astype(int)
#df2.insert(0, 'image_id', df2.pop('image_id'))
#df2.to_excel('df.xlsx', index=False)