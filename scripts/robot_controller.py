#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs 

from robotiq_msgs.msg import CModelCommand

from geometry_msgs.msg import Pose
from gesture_ur3.msg import Gesture

import numpy as np
import math
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import threading

# moveit_commander.roscpp_initialize(sys.argv)
# rospy.init_node('move_ur3', anonymous=True)

# robot = moveit_commander.RobotCommander()
commander = moveit_commander.MoveGroupCommander("arm")

# display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

# eef_link = commander.get_end_effector_link()
# # print ("End effector: %s") % eef_link
# group_names = robot.get_group_names()
# print ("Robot Groups:", robot.get_group_names())
# print ("Printing robot state:")
# print (robot.get_current_state())
# print ("")


# Joint configuration
commander.set_max_velocity_scaling_factor(0.1)
commander.set_max_acceleration_scaling_factor(0.1)

# Planner configuration
commander.set_planner_id("RRTConnect")
commander.set_planning_time(5)
commander.set_num_planning_attempts(10)

base_rotation_angle = math.pi*(3/4)     # fixed angle to rotate the base of the robot

# z_axis_correction = 0.24 + 0.05              # correction of the z-axis for vertical gripper
# z_axis_correction = 0.15 + 0.05              # correction of the z-axis for angled gripper (45deg)
z_axis_correction = 0.20 + 0.05              # correction of the z-axis for angled gripper (60deg)

class RobotController:
    def __init__(self):
        self.current_pose = Pose()
        # self.current_pose = commander.get_current_pose().pose
        # print("Current pose: \n", self.current_pose)
        # starting_pose = commander.get_current_pose().pose
        # starting_pose.position.x = 0.1317
        # starting_pose.position.y = 0.2940
        # starting_pose.position.z = 0.3166
        # starting_pose.orientation.x = -0.9213
        # starting_pose.orientation.y = 0.3888
        # starting_pose.orientation.z = 0.0
        # starting_pose.orientation.w = 0.007
        
        #-----------------INITIAL POSITION-----------------#
        starting_position = commander.get_current_joint_values()
        print("Starting position: ", starting_position)
        
        starting_position[0] = 0.785
        starting_position[1] = -1.5708
        starting_position[2] = 1.5708
        # starting_position[3] = -2.3562  # 45 degrees
        # starting_position[3] = -1.5708  # 90 degrees
        starting_position[3] = -2.0944  # 60 degrees
        starting_position[4] = -1.5708
        starting_position[5] = 0.005
        commander.go(starting_position, wait=True)
        commander.stop()

        # Robot's current and next (target) pose
        self.current_pose = starting_position
        self.next_pose = None

        #-----------------SUBSCRIBERS-----------------#
        rospy.Subscriber('coordinates', Pose, self.coord_callback)
        rospy.Subscriber('gestures', Gesture, self.gesture_callback)


        #-----------------ROBOTIQ GRIPPER-----------------#
        # Publisher
        self.command_pub = rospy.Publisher('command', CModelCommand, queue_size=3)

        # Initialize the command message
        self.current_command = CModelCommand()
        self.previous_command = CModelCommand()

        # Set the command message to the default values and publish it
        self.current_command.rACT = 1   # Activate the gripper
        self.current_command.rGTO = 1   # Go to position
        self.current_command.rSP = 255  # Speed
        self.current_command.rFR = 0    # Force
        self.command_pub.publish(self.current_command)

        # Gripper states (open/closed)
        self.previous_gripper_state = None
        self.current_gripper_state = None


        # Arrays for plot values
        self.current_poses = []
        self.pose_goals = []
        self.pose_diff = []

        # Flag for the callback function
        self.coord_callback_flag = True

        # Flag for the robot activation
        self.activate_robot_flag = False

        # Time delay for on/off switch
        self.last_toggle_time = time.time()
        self.toggle_delay = 3

        # Frequencies of robot movement [Hz]
        self.normal_move_rate = rospy.Rate(1)
        self.slow_move_rate = rospy.Rate(0.5)

        self.min_distance = 0.005
        self.max_distance = 0.5


    def plot_poses(self, current_poses, pose_goals, pose_differences):
        fig = plt.figure(figsize=(12, 6))

        # 3D plot of poses
        ax1 = fig.add_subplot(121, projection='3d')

        # Plot current_poses
        xs = [pose.position.x for pose in current_poses]
        ys = [pose.position.y for pose in current_poses]
        zs = [pose.position.z for pose in current_poses]
        ax1.plot(xs, ys, zs, c='r', linewidth = 1)
        ax1.scatter(xs[0], ys[0], zs[0], c='r', marker='o')  # Mark the beginning


        # Plot pose_goals
        xs = [pose.position.x for pose in pose_goals]
        ys = [pose.position.y for pose in pose_goals]
        zs = [pose.position.z for pose in pose_goals]
        ax1.plot(xs, ys, zs, c='b', linewidth = 1)
        ax1.scatter(xs[0], ys[0], zs[0], c='b', marker='o')  # Mark the beginning

        ax1.set_xlim([0, 0.4])
        ax1.set_ylim([0, 0.4])
        ax1.set_zlim([0.25, 0.45])

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 2D plot of differences
        ax2 = fig.add_subplot(122)

        ax2.plot(pose_differences, c='g')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Difference')

        plt.tight_layout()
        plt.show()

    def rotate_point_around_z(self, theta, x, y, z):
        """
        Rotate the coordinate system around Z axis
        
        This puts the robot workspace where I want it
        """
        # Convert angle from degrees to radians
        # theta = np.radians(angle)
        
        # Rotation matrix for rotation around the z-axis
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Original point vector
        point = np.array([x, y, z])
        
        # Rotated point vector
        rotated_point = R.dot(point)
        
        return rotated_point
    
    def calculate_distance(self, current_pose, next_pose):
        """
        Calculate and return an euclidean distance between two points in 3D space.
        """
        distance = math.sqrt((next_pose.position.x - current_pose.position.x)**2 + 
                             (next_pose.position.y - current_pose.position.y)**2 + 
                             (next_pose.position.z - current_pose.position.z)**2
                             )
        print("Calculated distance: ", distance)
        return distance

    def set_new_pose_goal(self, pose_goal, new_x, new_y, new_z):
        """
        Assign and return a new target pose for the robot.

        Also corrects the Z axis (line 49) 
        """
        pose_goal.position.x = -new_x
        pose_goal.position.y = -new_y
        pose_goal.position.z = new_z + z_axis_correction
        return pose_goal


    def coord_callback(self, data):
        """
        Process coordinate data from the topic.

        Read target coordinates, adjust the coordinate system and set a new pose goal
        """
        # rospy.loginfo(rospy.get_caller_id() + "%s", data)
        self.current_pose = commander.get_current_pose().pose
        # print("Current pose: \n", self.current_pose.position)
        pose_goal = commander.get_current_pose().pose
        rotated_point = self.rotate_point_around_z(base_rotation_angle, data.position.x, data.position.y, data.position.z)
        new_pose_goal = self.set_new_pose_goal(pose_goal, rotated_point[0], rotated_point[1], rotated_point[2])
        # print("Pose goal: \n", new_pose_goal.position)
        self.next_pose = new_pose_goal


        # if self.coord_callback_flag == True:
            # self.coord_callback_flag = False
        # if self.min_distance < self.calculate_distance(self.current_pose, self.next_pose) < self.max_distance:
        #     self.current_poses.append(self.next_pose)
        #     commander.set_pose_target(self.next_pose)
        #     # print(commander.plan(pose_goal))
        #     commander.go(self.next_pose, wait=True)
        #     commander.stop()
        #     commander.clear_pose_targets()

        #     # Plot stuff
        #     after_pose = commander.get_current_pose().pose
        #     difference = self.calculate_distance(self.next_pose, after_pose)
        #     self.pose_diff.append(difference)
        #     self.pose_goals.append(after_pose)
        # else:
        #     print("Distance is outside the acceptable range. Skipping the movement.")
            # self.coord_callback_flag = True
        
        # Naruszenie ochrony pamięci (zrzut pamięci)

    def gesture_callback(self, data):
        """
        Process gesture data from the topic.

        'Closed_Fist', 'Open_Palm' - gripper closed/open
        'Thumb_Up' - robot switched on/off 
        """

        # Robot on/off switch in form of a Thumb Up gesture
        # The switch is given a time delay to prevent constant switching
        current_time = time.time()
        if data.gesture == ['Thumb_Up'] and (current_time - self.last_toggle_time) > self.toggle_delay:
            self.activate_robot_flag = not self.activate_robot_flag
            self.last_toggle_time = current_time  # Last toggle time update
            if self.activate_robot_flag:
                print("Robot activated.")
            else: 
                print("Robot deactivated.")

        if self.activate_robot_flag:
            if data.gesture == ['Closed_Fist']:
                self.current_command.rPR = 255
                print("Command rPR: ", self.current_command.rPR)
                self.current_gripper_state = 'closed'
            if data.gesture == ['Open_Palm']:
                self.current_command.rPR = 0
                print("Command rPR: ", self.current_command.rPR)
                self.current_gripper_state = 'open'
            if data.gesture == ['None']:
                print("No gesture detected.")
            self.previous_command = self.current_command
            print("Current force: ", self.current_command.rFR)
            self.command_pub.publish(self.current_command)


    def move_robot(self):
        """
        A thread with the main robot control loop.
        """
        while not rospy.is_shutdown():
            if self.previous_gripper_state == 'open' and self.current_gripper_state == 'closed':
                # Hold the robot still for 1 second
                rospy.sleep(1)
            self.previous_gripper_state = self.current_gripper_state

            if self.next_pose == None:
                print("No new pose goal.")
                rospy.sleep(0.1)
                continue

            if not self.activate_robot_flag:
                rospy.sleep(1)
                continue

            print("Next pose: \n", self.next_pose)
            if self.min_distance < self.calculate_distance(self.current_pose, self.next_pose) < self.max_distance:
                self.current_poses.append(self.next_pose)
                commander.set_pose_target(self.next_pose)
                # print(commander.plan(pose_goal))
                commander.go(self.next_pose, wait=True)
                commander.stop()
                commander.clear_pose_targets()

                # Plot stuff
                after_pose = commander.get_current_pose().pose
                difference = self.calculate_distance(self.next_pose, after_pose)
                self.pose_diff.append(difference)
                self.pose_goals.append(after_pose)
            else:
                print("Distance is outside the acceptable range. Skipping the movement.")
        
            self.normal_move_rate.sleep()
    
    def robot_controller(self):
        rospy.spin()
        # self.plot_poses(self.current_poses, self.pose_goals, self.pose_diff)

if __name__ == '__main__':
    try:
        rospy.init_node('robot_controller')

        rc = RobotController()
        move_worker = threading.Thread(target=rc.move_robot)
        move_worker.start()
        # rc.move_robot()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
