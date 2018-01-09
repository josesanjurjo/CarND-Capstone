#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
SLOWDOWN = 0.2


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.current_pose = None
        self.base_waypoints = None
        self.base_waypoint_search = None
        self.velocity = 0.
        self.stop_wp = -1

        rospy.spin()

    def pose_cb(self, msg):
        self.current_pose = msg.pose
        self.frame_id = msg.header.frame_id
        self.update()

    def waypoints_cb(self, msg):
        rospy.loginfo("WaypointUpdater: Got Base Waypoints")
        self.base_waypoints = msg.waypoints
        self.base_waypoint_search = msg.waypoints + msg.waypoints

    def velocity_cb(self, msg):
        self.velocity = msg.twist.linear.x

    def traffic_cb(self, msg):
        rospy.loginfo("WaypointUpdater: Red light in WayPoint #{}".format(msg.data))
        self.stop_wp = msg.data

    def obstacle_cb(self, msg):
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position,
                       waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def update(self):
        """ Updates the waypoints and publishes them to /final_waypoints
        """

        required = (
            self.current_pose,
            self.base_waypoints
        )
        if any(x is None for x in required):
            return

        base_waypoint_idx = self.nearest_waypoint()

        next_waypoints = self.base_waypoint_search[
                         base_waypoint_idx:base_waypoint_idx + LOOKAHEAD_WPS]

        stop_is_close = self.is_stop_close(base_waypoint_idx)
        if stop_is_close:
            for i, waypoint in enumerate(next_waypoints):
                waypoint.twist.twist.linear.x = self.brake(
                    base_waypoint_idx + i)

        lane = Lane()
        lane.header.frame_id = self.frame_id
        lane.waypoints = next_waypoints
        lane.header.stamp = rospy.Time.now()
        self.final_waypoints_pub.publish(lane)

    def is_stop_close(self, base_waypoint_idx):
        """ Checks whether it is time to start slowing down
        """
        stop_is_close = False
        if self.stop_wp - base_waypoint_idx > 0:  # stop is ahead
            d_stop = self.distance(
                self.base_waypoints, base_waypoint_idx, self.stop_wp)
            current_wp = self.base_waypoints[base_waypoint_idx]
            stop_is_close = d_stop < current_wp.twist.twist.linear.x ** SLOWDOWN
        return stop_is_close

    def is_behind(self, nearest_idx):
        """  Check if nearest_idx is behind the car
        """
        # TODO: Implement
        return False

    def nearest_waypoint(self):
        """ Finds the nearest base waypoint to the current pose
        """
        position = self.current_pose.position
        nearest_d = float('inf')
        nearest_idx = None

        for i, wp in enumerate(self.base_waypoints):
            d = self.raw_distance(position, wp.pose.pose.position)
            if d < nearest_d:
                nearest_idx, nearest_d = i, d

        if self.is_behind(nearest_idx):
            nearest_idx += 1

        return nearest_idx

    def brake(self, i):
        """ Decreases waypoint velocity
        """
        wp = self.base_waypoints[i]
        wp_speed = wp.twist.twist.linear.x

        d_stop = self.distance(self.base_waypoints, i, self.stop_wp)

        speed = 0.
        if d_stop > 0:
            speed = d_stop * (wp_speed ** (1. - SLOWDOWN))
        if speed < 1:
            speed = 0.

        return speed

    @staticmethod
    def raw_distance(a, b):
        x = a.x - b.x
        y = a.y - b.y
        return x ** 2 + y ** 2


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
