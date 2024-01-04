"""
python3 predict_from_video.py
"""

import os
import cv2
import PIL
import torchvision.transforms as transforms
import numpy as np

from gestures import GESTURES
from one_stage_detector import OneStageDetector
from sliding_window import SlidingWindow
from overlap_tracking_hand import CentroidTracker


IN_VIDEO_PATH = os.path.join("in_video.avi")
OUT_VIDEO_PATH = os.path.join("out_video.avi")
DETECTOR_THRESH = 0.5   #0.5
SIZE_THRESH = None
FPS = 30
MAX_DISAPPEARED = FPS               # number of frames after wich the tracker consider an unseen entity disappeared
WINDOW_SIZE = FPS


transforms = transforms.Compose([transforms.ToTensor()]) 
detector = OneStageDetector(conf_thresh=DETECTOR_THRESH, size_thresh=SIZE_THRESH)
hands_tracker = CentroidTracker(maxDisappeared=MAX_DISAPPEARED)
sliding_windows = dict()

in_video = cv2.VideoCapture(IN_VIDEO_PATH)
width  = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))   # float 'width'
height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float 'height'
fps = int(in_video.get(cv2.CAP_PROP_FPS))              # float 'fps'
print("Property of input video:\nwidth: {}\theight: {}\tfps: {}".format(width, height, fps))
#width = 852
out_video = cv2.VideoWriter(filename=OUT_VIDEO_PATH, fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), fps=fps, frameSize=(width, height))
#out_video = cv2.VideoWriter(filename=OUT_VIDEO_PATH, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=fps, frameSize=(width, height))    # for mp4 file
 
# Read until video is completed
while(in_video.isOpened()):
  # Capture frame-by-frame
  ret, frame = in_video.read()
  if ret == True:
    # Preprocess image
    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(cv_image)
    img = transforms(pil_image)

    # Predict gesture
    hands = detector.detect(img)
    hand_bboxes = list()
    left = hands['roi'][0]
    top = hands['roi'][1]
    right = hands['roi'][2]
    bottom = hands['roi'][3]
    x_center = (right + left) // 2
    y_center = (bottom + top) // 2
    hand_bboxes.append(hands['roi'])
    centr_hands_in_frame, window_hands_in_frame = hands_tracker.update(rects=hand_bboxes)

    # Make labeled frame
    gesture, confidence = 0, 0.
    (objectID, (centroid_x, centroid_y)) = next(iter(centr_hands_in_frame.items()))
    if centroid_x==x_center and centroid_y==y_center:
        probabilty_vector = np.zeros(len(GESTURES))
        label_predicted = int(hands['label'])
        print(GESTURES[label_predicted]['eng'])
        score_prediction = float(hands['confidence'])
        probabilty_vector[label_predicted] = score_prediction
        

        ## add prediction to the sliding window
        if objectID not in sliding_windows:
            sliding_windows[objectID] = SlidingWindow(window_size=WINDOW_SIZE, element_size=len(GESTURES))
            sliding_windows[objectID].put(probabilty_vector)
        else:
            sliding_windows[objectID].put(probabilty_vector)
        
        ## apply majority voting on sliding window for gesture classification
        gesture, confidence = sliding_windows[objectID].get_max()

        # write on the frame
        (objectID, (centroid_x, centroid_y)) = next(iter(centr_hands_in_frame.items()))
        cv2.rectangle(img=frame, pt1=(int(left), int(top)), pt2=(int(right), int(bottom)), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, GESTURES[gesture]['eng'], (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * 3 * 200, (0, 0, 255), 2)

    # Save labeled frame
    cv2.imshow("Detections", frame)
    out_video.write(frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break
 
# When everything done, release the video capture objects
in_video.release()
out_video.release()

# Closes all the frames
cv2.destroyAllWindows()