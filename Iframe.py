import cv2
import os
folder_path = "F:\object-detection-opencv-master\Frames"
files = os.listdir(folder_path)
for file in files:
    if file.endswith(('.jpg')):
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
cv2.destroyAllWindows()


