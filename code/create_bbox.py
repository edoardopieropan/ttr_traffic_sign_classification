'''
Permette il savataggio delle bounding-box dal video. L'aggiunta di una bbox si effettua premendo il tasto 'n'.
'''

import cv2
 
# Read video
video = cv2.VideoCapture("../video/test_video.mp4")

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

with open("../utils/bbox.txt","w+") as f:
    frame_counter=0
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        frame_counter+=1
        cv2.imshow("Tracking", frame)

        n = cv2.waitKey(100) & 0xff

        # Add bbox
        if n==110:
            bbox = cv2.selectROI(frame, False)
            cv2.destroyWindow('ROI selector')
            f.write(str(frame_counter)+" "+str(bbox)+"\n")

        # Exit if ESC pressed
        else if n == 27 : break