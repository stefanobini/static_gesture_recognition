#!/usr/bin/env python3
from charset_normalizer import detect
from cv_bridge import CvBridge
import rospy
import time
import cv2
import numpy as np
import os
import uuid
import PIL
from colorama import Fore
import torchvision.transforms as transforms 

from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
# from demo_utils.ai.video.hand_detection import HandDetector

from commands import GESTURES
# from settings import demo_settings
from demo_utils.gesture_recognition import SlidingWindow
from demo_utils.overlap_tracking_hand import CentroidTracker
from static_gesture_recognition.mobilenetv3ssd.predict_from_video.one_stage_detector import OneStageDetector
from demo_utils.post_request import MyRequestPost
from settings.io import io


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_sqmalloc_async'
WINDOW_SIZE = io.camera.fps                                         # size of the sliding windows to average the gesture classification
N_GESTURES = len(GESTURES)                      # number of gesture considered
MAX_DISAPPEARED = io.camera.fps                                     # number of frames after wich the tracker consider an unseen entity disappeared
DETECTOR_THRESH = 0.5
SIZE_THRESH = None
ROBOT_UUID = uuid.uuid1().node
GESTURE_THRESHOLD:float = 0.3


class Callback:

    
    def __init__(self, publisher=None, detector=None, post_request=None, webviewer=False):
        self.bridge = CvBridge()
        self.detector = detector
        self.publisher = publisher
        self.post_request = post_request
        self.hands_tracker = CentroidTracker(maxDisappeared=MAX_DISAPPEARED)
        self.window_size = WINDOW_SIZE
        self.n_gestures = N_GESTURES
        self.webviewer = webviewer
        self.sliding_windows = dict()
        self.frame = 0
        self.timestamp = time.time()
        self.transforms = transforms.Compose([transforms.ToTensor()]) 
        self.last_gesture = 0


    def __call__(self, msg):
        #prevoius_timestamp = self.timestamp
        response = Detection2DArray()
        response.header.stamp = msg.header.stamp
        img_bgr = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data) # RAW image with shape (height, width, channel)
        pil_image = None
        try:
            #cv_image = self.bridge.imgmsg_to_cv2(img_bgr)
            cv_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(cv_image)
            img = self.transforms(pil_image)
        except CvBridgeError as e:
            print(e)
        img_printed = cv_image
        #cv2.imwrite("/home/felice/command_interaction/ROS/hri_ws/acquisition_test_bgr.png", img_printed)

        # rospy.loginfo(img.shape)
        
        hands = self.detector.detect(img)

        #print(hands['label'], GESTURES[hands['label']]["eng"])
        #hands = self.detector.onnx_cpu_detect(img_rgb)

        hand_bboxes = list()
        x_center = (hands['roi'][2] + hands['roi'][0]) // 2
        y_center = (hands['roi'][3] + hands['roi'][1]) // 2
        hand_bboxes.append(hands['roi'])
        centr_hands_in_frame, window_hands_in_frame = self.hands_tracker.update(rects=hand_bboxes)
        x_center = (hands['roi'][2] + hands['roi'][0]) // 2
        y_center = (hands['roi'][3] + hands['roi'][1]) // 2
        
        #"""
        gesture, confidence = 0, 0.
        (objectID, (centroid_x, centroid_y)) = next(iter(centr_hands_in_frame.items()))
        if centroid_x==x_center and centroid_y==y_center:
            probabilty_vector = np.zeros(self.n_gestures)
            label_predicted = int(hands['label'])
            score_prediction = float(hands['confidence'])
            probabilty_vector[label_predicted] = score_prediction
            

            ## add prediction to the sliding window
            if objectID not in self.sliding_windows:
                self.sliding_windows[objectID] = SlidingWindow(window_size=self.window_size, element_size=self.n_gestures)
                self.sliding_windows[objectID].put(probabilty_vector)
            else:
                self.sliding_windows[objectID].put(probabilty_vector)
            
            ## apply majority voting on sliding window for gesture classification
            gesture, confidence = self.sliding_windows[objectID].get_max()

            (objectID, (centroid_x, centroid_y)) = next(iter(centr_hands_in_frame.items()))
            ## build classification message
            detection = Detection2D()
            detection.header.frame_id = str(gesture)
            detection.header.stamp = msg.header.stamp
            detection.bbox.size_x = window_hands_in_frame[objectID][0]
            detection.bbox.size_y = window_hands_in_frame[objectID][1]
            detection.bbox.center.x = centroid_x
            detection.bbox.center.y = centroid_y
            result = ObjectHypothesisWithPose()
            result.id = objectID
            result.score = confidence
            detection.results.append(result)
            # detection.source_img =self.bridge.cv2_to_imgmsg(img)
        #"""
        #cv2.imwrite(filename="/home/felice/command_interaction/ROS/detected_frames/gesture{:02d}_frame_{}.png".format(gesture, str(self.frame)[-1:]), img=img_printed)
        self.frame += 1
        
        # self.publisher.publish(response)      # IF WE WANT PUBLISH ON WEBVIEWER TO SEE THE RESULTS ON PEPPER?S TABLET
        gesture = int(detection.header.frame_id)
        confidence = float(detection.results[0].score)
        #print(gesture, GESTURES[gesture]["eng"])
        if gesture!=0 and gesture!=N_GESTURES-1 and confidence > GESTURE_THRESHOLD:
            if self.post_request is not None:
                # self.post_request.send_command(command_id=response.detections[0].header.frame_id, confidence=response.detections[0].results[0].score)    # IF WE WANT PUBLISH ON FIWARE CONTEXTBROKER
                self.post_request.send_command(command_id=gesture, confidence=confidence)    # IF WE WANT PUBLISH ON FIWARE CONTEXTBROKER
            elif gesture!=self.last_gesture:
                res_str = Fore.MAGENTA + '#'*6 + ' GESTURE FRAME  ' + '#'*6 + '\n# ' + Fore.LIGHTMAGENTA_EX + '{}: {:.3f}'.format(GESTURES[gesture]["eng"], confidence) + Fore.MAGENTA + ' #\n# ' + Fore.LIGHTMAGENTA_EX + '{}: {:.3f}'.format(GESTURES[gesture]["ita"], confidence) + Fore.MAGENTA + ' #\n' + '#'* 44 + Fore.RESET + '\n'
                print(res_str)

        if self.webviewer:
            # self.publisher.publish(response)
            self.publisher.publish(detection)
        
        self.last_gesture = gesture

        #self.timestamp = time.time()
        #det_time = self.timestamp - prevoius_timestamp    
        #frame_rate = round(1 / (det_time), 2)
        #print("FPS:\t{:.1f}\tTime:\t{:.4f} s".format(frame_rate, det_time))    # -> {FPS: 8;    Time: 0.125 s}
        


class HandDetectorNode:


    def start(self):
        global ROBOT_UUID

        rospy.init_node('hand_gesture_recognition_node', anonymous=True)
        # pub = rospy.Publisher("hand_gesture_recognition", Detection2DArray, queue_size=1)
        pub = rospy.Publisher("hand_gesture_recognition", Detection2D, queue_size=1)
        
        detector = OneStageDetector(conf_thresh=DETECTOR_THRESH, size_thresh=SIZE_THRESH)

        LANGUAGE = rospy.get_param("/language")
        FIWARE_CB = rospy.get_param("/fiware_cb")
        WEBVIEWER = rospy.get_param("/webviewer")

        if FIWARE_CB != "None":
            post_request = MyRequestPost(instance_uuid=ROBOT_UUID, entity="UNISA.SpeechGestureAnalysis.Gesture", msg_type="Gesture", address=FIWARE_CB, port=1026)
            post_request.create_entity()
            callback = Callback(publisher=pub, detector=detector, post_request=post_request)
        if WEBVIEWER:
            callback = Callback(publisher=pub, detector=detector, webviewer=WEBVIEWER)
        
        print(Fore.GREEN + '#################\n#               #\n# MODELS LOADED #\n#               #\n#################' + Fore.RESET)
        sub = rospy.Subscriber("in_rgb", Image, callback)

        rospy.spin()


if __name__ == "__main__":
    try:
        node = HandDetectorNode()
        node.start()
    except rospy.ROSInterruptException:
        pass
