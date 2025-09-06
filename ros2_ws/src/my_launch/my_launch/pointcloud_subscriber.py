import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('pointcloud_subscriber')
        topic = '/topic_2/cam_2/depth/color/points'
        self.subscription = self.create_subscription(
            PointCloud2,
            topic,  # ← Change this to match your actual pointcloud topic
            self.pointcloud_callback,
        self.get_logger().info(f'PointCloud2 subscriber to topic: {topic}')
        )

    def pointcloud_callback(self, msg):
        # Convert the point cloud to a list of points
        point_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        point_list = list(point_generator) # Here is where the data is stored
        # self.get_logger().info(f"Received PointCloud2 — width: {msg.width}, height: {msg.height}")

        self.get_logger().info(f'Received point cloud with {len(point_list)} points')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()