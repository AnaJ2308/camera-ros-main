#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class MultiCameraSubscriber(Node):
    def __init__(self):
        super().__init__('multi_camera_subscriber')
        self.latest_points = {}
        # List of up to 4 camera topics
        self.camera_topics = [
            '/topic_1/cam_1/depth/color/points',
            '/topic_2/cam_2/depth/color/points',
            '/topic_3/cam_3/depth/color/points',
            '/topic_4/cam_4/depth/color/points'
        ]
        
        for topic in self.camera_topics:
            self.create_subscription(
                PointCloud2,
                topic,
                lambda msg, t=topic: self.pointcloud_callback(msg, t),
                10
            )
            self.get_logger().info(f'Subscribed to {topic}')

    def pointcloud_callback(self, msg, topic_name):
        # Convert PointCloud2 to list of points
        point_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        point_list = list(point_generator) # where data is stored
        self.latest_points[topic_name] = point_list

        self.get_logger().info(
            f'Received point cloud from {topic_name} with {len(point_list)} points'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()