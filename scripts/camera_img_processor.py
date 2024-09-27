#!/usr/bin/env python

import sys
import rospy

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from gesture_ur3.msg import Gesture

from cv_bridge import CvBridge, CvBridgeError

import os
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
import threading 
import datetime as dt

class GestureRecognizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.org = (20, 40)
        self.fontScale = .75
        self.color = (255,255,255)
        self.thickness = 1
        
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("camera_image",Image,queue_size=10)
        self.gesture_pub = rospy.Publisher("gestures", Gesture, queue_size=10)
        self.coordinates_pub = rospy.Publisher("coordinates", Pose, queue_size=1)
        self.coord_rate = rospy.Rate(2)
        self.rate = rospy.Rate(30)

        self.coordinates2publish = Pose()
        self.current_gestures = []

    def main(self):

        # ====== Initialize Camera ======
        realsense_ctx = rs.context()
        connected_devices = [] # List of serial numbers for present cameras
        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            print(f"{detected_camera}")
            connected_devices.append(detected_camera)
        device = connected_devices[0] # In this example we are only using one camera
        pipeline = rs.pipeline()
        config = rs.config()
        background_removed_color = 153 # Grey

        # For better FPS. but worse resolution:
        stream_res_x = 640
        stream_res_y = 480
        stream_fps = 30
        config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
        config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
        profile = pipeline.start(config)
        align_to = rs.stream.color

        # ====== Get depth Scale ======
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

        # ====== Set clipping distance ======
        clipping_distance_in_meters = 2
        clipping_distance = clipping_distance_in_meters / depth_scale
        print(f"\tConfiguration Successful for SN {device}")
        align = rs.align(align_to)

        # MediaPipe task configuration
        num_hands = 1
        model_path = os.path.join(os.path.dirname(__file__), 'gesture_recognizer.task')
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.lock = threading.Lock()
        self.current_gestures = []
        options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands = num_hands,
            result_callback=self.__result_callback)
        recognizer = GestureRecognizer.create_from_options(options)

        timestamp = 0 
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65)

        # cap = cv2.VideoCapture(0)

        while True:
            # ret, frame = cap.read()
            # if not ret:
            #     break
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # results = hands.process(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = dt.datetime.today().timestamp() # Necessary for FPS calculations

            # Get and align frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            # Process images
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_image_flipped = cv2.flip(depth_image,1)
            color_image = np.asanyarray(color_frame.get_data())
            # depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
            # background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # images = cv2.flip(background_removed,1)
            color_image = cv2.flip(color_image,1)
            images = color_image
            color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(images, "bgr8"))
            except CvBridgeError as e:
                print(e)



            results = hands.process(color_images_rgb)
            i=0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(images, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_images_rgb)
                    recognizer.recognize_async(mp_image, timestamp)
                    timestamp = timestamp + 1 # should be monotonically increasing, because in LIVE_STREAM mode

                    hand_side_classification_list = results.multi_handedness[i]
                    hand_side = hand_side_classification_list.classification[0].label
                    # print(f"Hand side: {hand_side}")

                    wrist = results.multi_hand_landmarks[i].landmark[0]
                    x = int(wrist.x*len(depth_image_flipped[0]))
                    y = int(wrist.y*len(depth_image_flipped))
                    if x >= len(depth_image_flipped[0]):
                        x = len(depth_image_flipped[0]) - 1
                    if y >= len(depth_image_flipped):
                        y = len(depth_image_flipped) - 1
                    mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
                    z = mfk_distance
                    # print("x:{}, y:{}, z:{}".format(x, y, z))
                    x_coord = -(x - 0.5*stream_res_x)
                    y_coord = -(y - 0.5*stream_res_y)

                    x_coord = int((x_coord*z)/0.5)
                    y_coord = int((y_coord*z)/0.5)

                    if -stream_res_x/2 < x_coord < stream_res_x/2 and -stream_res_y/2 < y_coord < stream_res_y/2 and 0.5 < z < 1.0:
                        x_scaled = self.scale_range([x_coord], [-stream_res_x/2, stream_res_x/2], [-0.4, 0.4])
                        y_scaled = self.scale_range([z], [0.5, 1.0], [-0.1, 0.5])
                        z_scaled = self.scale_range([y_coord], [-stream_res_y/2, stream_res_y/2], [0.1, 0.5])

                        self.coordinates2publish.position.x = x_scaled[0]
                        self.coordinates2publish.position.y = y_scaled[0]
                        self.coordinates2publish.position.z = z_scaled[0]
                    else:
                        print('Hand out of bounds')
                    
                    # images = cv2.line(images, (160, 110), (160, 330), self.color, 3)
                    # images = cv2.line(images, (160, 330), (480, 330), self.color, 3)
                    # images = cv2.line(images, (480, 330), (480, 110), self.color, 3)
                    # images = cv2.line(images, (480, 110), (160, 110), self.color, 3)
                    org2 = (20, self.org[1]+(20*(i+1)))
                    images = cv2.putText(images, f"{hand_side} Hand: x:{x_coord} y:{y_coord} z:{z:0.3}", org2, self.font, self.fontScale, [0,0,0], self.thickness+1, cv2.LINE_AA)
                    images = cv2.putText(images, f"{hand_side} Hand: x:{x_coord} y:{y_coord} z:{z:0.3}", org2, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
                    i+=1
                self.put_gestures(images)

            # Display FPS
            time_diff = dt.datetime.today().timestamp() - start_time
            fps = int(1 / time_diff)
            # images = cv2.putText(images, f"FPS: {fps}", self.org, self.font, self.fontScale, [0,0,0], self.thickness+1, cv2.LINE_AA)
            # images = cv2.putText(images, f"FPS: {fps}", self.org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', images)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break
            self.rate.sleep()

        # cap.release()

    def coordinate_publisher_thread(self):
        while not rospy.is_shutdown():
            coord_msg = self.coordinates2publish
            self.coordinates_pub.publish(coord_msg)
            # gesture_msg = Gesture()
            # gesture_msg.gesture = self.current_gestures
            # self.gesture_pub.publish(gesture_msg)
            self.coord_rate.sleep()

    def gesture_publisher_thread(self):
        while not rospy.is_shutdown():
            gesture_msg = Gesture()
            gesture_msg.side = ['right']
            gestures = self.current_gestures
            gesture_msg.gesture = gestures
            # print(type(gestures))
            # rospy.loginfo(f"Publishing gestures: {gestures}")
            self.gesture_pub.publish(gesture_msg)
            self.coord_rate.sleep()

    def put_gestures(self, frame):
        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        org_g = [20, 100]
        for hand_gesture_name in gestures:
            # show the prediction on the frame
            cv2.putText(frame, hand_gesture_name, org_g, self.font, 
                                self.fontScale, [0,0,0], self.thickness+1, cv2.LINE_AA)
            cv2.putText(frame, hand_gesture_name, org_g, self.font, 
                                self.fontScale, self.color, self.thickness, cv2.LINE_AA)
            org_g[1]+=20

    def __result_callback(self, result, output_image, timestamp_ms):
        #print(f'gesture recognition result: {result}')
        self.lock.acquire() # solves potential concurrency issues
        self.current_gestures = []
        if result is not None and any(result.gestures):
            # print("Recognized gestures:")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                print(gesture_name)
                self.current_gestures.append(gesture_name)
        self.lock.release()

    def scale_range(self, numbers, original_range, target_range):
        original_min, original_max = original_range
        target_min, target_max = target_range
        
        # Calculate the scale factor
        scale_factor = (target_max - target_min) / (original_max - original_min)
        
        # Apply the transformation to each number in the list
        scaled_numbers = [(target_min + (x - original_min) * scale_factor) for x in numbers]
        
        return scaled_numbers

if __name__ == "__main__":
    rospy.init_node('camera_image_processor')

    rec = GestureRecognizer()
    c_worker = threading.Thread(target=rec.coordinate_publisher_thread)
    c_worker.start()
    g_worker = threading.Thread(target=rec.gesture_publisher_thread)
    g_worker.start()
    rec.main()
    cv2.destroyAllWindows()
    cv2.waitKey(1)