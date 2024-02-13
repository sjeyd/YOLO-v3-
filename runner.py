import os
import scipy
import cv2 as cv
from objectframe import run_object_detection
# Paths to YOLO config, weights, and classes files
config_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
classes_path = 'yolov3.txt'
All_data_dicts = []
# Directory containing images
images_directory = './Frames/Video01'
output_path = "./Outputimage/Video01"
for count, item in enumerate(os.listdir(images_directory)):
    if count % 10 == 1:
        item_path = os.path.join(images_directory, item)
        if os.path.isfile(item_path) and item.endswith(".jpg"):
            data_dict = run_object_detection(item_path, config_path, weights_path, classes_path, output_path,count)
            # save frames
            #cv.imwrite("F:\\object-detection-opencv-master\\Outputimage\\Object_Frame%d.jpg" % i , image)
            ###
            #items_with_i = [(count, *item_values) for item_values in data_dict.get('items', [])]
            items_with_i=[count,data_dict]
            All_data_dicts.append(items_with_i)
            print('Iframe %d saved'% count)
        else:
            print('Not Iframe')
# Save all_data_dicts to a single .mat file

#print(All_data_dicts)
file_name = 'Bbox_video01.mat'
scipy.io.savemat(file_name, {'all_data_dicts': All_data_dicts}, appendmat=True, format='5')