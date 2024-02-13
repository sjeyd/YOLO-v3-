import os
import scipy
import cv2 as cv
from object_detect_fun import run_object_detection

# Paths to YOLO config, weights, and classes files


config_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
classes_path = 'yolov3.txt'
All_data_dicts = []
# Directory containing images
images_directory = './Frames'

for i, item in enumerate(os.listdir(images_directory)):
    if i % 10 == 1:
        item_path = os.path.join(images_directory, item)
        if os.path.isfile(item_path) and item.endswith(".jpg"):
            data_dict = run_object_detection(item_path, config_path, weights_path, classes_path)
            # save frames
            #cv.imwrite("F:\\object-detection-opencv-master\\Outputimage\\Object_Frame%d.jpg" % i , image)
            ###
            items_with_i = [(i, *item_values) for item_values in data_dict.get('items', [])]
            All_data_dicts.extend(items_with_i)
            
        else:
            print('Not Iframe')

# Save all_data_dicts to a single .mat file
print(All_data_dicts)
file_name = 'all_results_data.mat'
scipy.io.savemat(file_name, {'all_data_dicts': All_data_dicts}, appendmat=True, format='5')
