import cv2

def captureframe(video):
    video = cv2.VideoCapture(video)
    count = 1
    success = True
    # for count in range(1, 101):  # Assuming you want to capture 100 frames
    while success:
        success, image = video.read()
        if success:
            cv2.imwrite('F:\object-detection-opencv-master\Frames\Video01\Frame%d.jpg' % count, image)
            print('Frame %d saved' % count)
            count += 1
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    captureframe('VID1.mp4')

 
 