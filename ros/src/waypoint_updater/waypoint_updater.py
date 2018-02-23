#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
import tf
import math
import itertools
import time
import copy

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

LOOKAHEAD_WPS = 150  # Number of waypoints we will publish. You can change this number
SLOWDOWN = 0.2
PUBLISHING_RATE = 20  # per second
RECALCULATE_STEPS = 200  # Number of steps to do a full position calculation


class WaypointUpdater(object):
    def __init__(self):
        self.current_pose = None
        self.base_waypoints = None
        self.velocity = 0.
        self.stop_wp = -1
        self.car_yaw = None
        self.nearest_waypoint_idx = None

        _counter = itertools.count(1)
        self.recalculate_idx = lambda: (next(_counter) % RECALCULATE_STEPS == 0)

        self.stop_m = rospy.get_param('~stop_m', 16.)  # m ahead of stop line for the car to stop

        rospy.init_node('waypoint_updater')

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.nearest_waypoint_pub = rospy.Publisher('nearest_waypoint', Int32, queue_size=1)

        rate = rospy.Rate(PUBLISHING_RATE)
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()

    def pose_cb(self, msg):
        self.current_pose = msg.pose
        self.frame_id = msg.header.frame_id
        orientation = self.current_pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.car_yaw = euler[2]
        # self.update()

    def waypoints_cb(self, msg):
        rospy.loginfo("WaypointUpdater: Got Base Waypoints")
        self.base_waypoints = msg.waypoints
        self.base_waypoint_velocities = [wp.twist.twist.linear.x
                                         for wp in self.base_waypoints]
        self.base_waypoints_sub.unregister()

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
        if not self.current_pose or not self.base_waypoints:
            return

        self.nearest_waypoint_idx = self.nearest_waypoint()
        base_waypoint_idx = self.nearest_waypoint_idx
        self.nearest_waypoint_pub.publish(base_waypoint_idx)

        self.restore_base_velocities(base_waypoint_idx)

        # Wrap around the list, since the car may be towards the end of the list
        next_waypoints = [
            self.base_waypoints[i % len(self.base_waypoints)]
            for i in range(base_waypoint_idx, base_waypoint_idx + LOOKAHEAD_WPS)
        ]

        stop_is_close = self.is_stop_close(base_waypoint_idx)
        if stop_is_close:
            for i, waypoint in enumerate(next_waypoints):
                waypoint.twist.twist.linear.x = self.brake(
                    (base_waypoint_idx + i) % len(self.base_waypoints))

        lane = Lane()
        lane.header.frame_id = self.frame_id
        lane.waypoints = next_waypoints
        lane.header.stamp = rospy.Time.now()
        self.final_waypoints_pub.publish(lane)

    def restore_base_velocities(self, base_waypoint_idx):
        # Wrap around the list, since the car may be towards the end of the list
        for i in range(base_waypoint_idx, base_waypoint_idx + LOOKAHEAD_WPS):
            idx = i % len(self.base_waypoints)
            wp = self.base_waypoints[idx]
            v = self.base_waypoint_velocities[idx]
            wp.twist.twist.linear.x = v

    def is_stop_close(self, base_waypoint_idx):
        """ Checks whether it is time to start slowing down
        """
        stop_is_close = False

        if self.stop_wp > 0:
            # stop is ahead
            d_stop = self.distance(
                self.base_waypoints, base_waypoint_idx, self.stop_wp) - self.stop_m
            current_wp = self.base_waypoints[base_waypoint_idx]
            stop_is_close = d_stop < current_wp.twist.twist.linear.x ** SLOWDOWN

        return stop_is_close

    def is_behind(self, nearest_idx):
        """  Check if nearest_idx is behind the car
        """
        yaw = self.car_yaw
        nearest_wp_x = self.base_waypoints[nearest_idx].pose.pose.position.x
        nearest_wp_y = self.base_waypoints[nearest_idx].pose.pose.position.y
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        loc_x = (nearest_wp_x - x) * math.cos(yaw) + (nearest_wp_y - y) * math.sin(yaw)
        if loc_x < 0.0:
            return True
        return False

    def nearest_waypoint(self):
        """ Finds the nearest base waypoint to the current pose
        """
        position = self.current_pose.position
        nearest_d = float('inf')
        nearest_idx = None
        search_wp = 20

        if self.nearest_waypoint_idx is not None and not self.recalculate_idx():
            _wp = self.nearest_waypoint_idx
            search_list = [  # Wraps around the end of base_waypoints
                self.base_waypoints[i % len(self.base_waypoints)]
                for i in range(_wp, _wp + search_wp)
            ]
            enum = enumerate(search_list, _wp)
        else:
            enum = enumerate(self.base_waypoints)

        for i, wp in enum:
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

        d_stop = self.distance(self.base_waypoints, i, self.stop_wp) - self.stop_m

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
