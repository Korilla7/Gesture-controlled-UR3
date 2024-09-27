#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose


def publisher():
    # Initialize the ROS node
    rospy.init_node('test_publisher', anonymous=True)

    # Create a publisher object
    pub = rospy.Publisher('test_publisher', Pose, queue_size=10)

    # Set the publishing rate (in Hz)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        # Create a message object
        msg = Pose()
        msg.position.x = 0.1
        msg.position.y = 0.2
        msg.position.z = 0.3

        # Publish the message
        pub.publish(msg)

        # Sleep to maintain the publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass