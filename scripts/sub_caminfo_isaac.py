#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import numpy as np

class CameraInfoSubscriber(Node):

    def __init__(self):
        super().__init__('camera_info_subscriber')
        self.subscription = self.create_subscription(
            CameraInfo,
            '/tb_1_green/camera_info',
            self.camera_info_callback,
            10)
        self.subscription  # prevent unused variable warning

    def camera_info_callback(self, msg):
        # Extract K matrix (3x3)
        K = np.array(msg.k).reshape(3, 3)
        
        # Extract P matrix (3x4)
        P = np.array(msg.p).reshape(3, 4)
        
        # Extract R matrix (3x3)
        R = np.array(msg.r).reshape(3, 3)

        # Print the matrices
        self.get_logger().info('Received camera info:')
        self.get_logger().info(f'K matrix:\n{K}')
        self.get_logger().info(f'P matrix:\n{P}')
        self.get_logger().info(f'R matrix:\n{R}')

        # Extract specific parameters
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # Extract translation components from P matrix
        Tx = P[0, 3]
        Ty = P[1, 3]

        self.get_logger().info(f'Focal length (fx, fy): ({fx}, {fy})')
        self.get_logger().info(f'Principal point (cx, cy): ({cx}, {cy})')
        self.get_logger().info(f'Translation (Tx, Ty): ({Tx}, {Ty})')

def main(args=None):
    rclpy.init(args=args)

    camera_info_subscriber = CameraInfoSubscriber()

    rclpy.spin(camera_info_subscriber)

    camera_info_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()