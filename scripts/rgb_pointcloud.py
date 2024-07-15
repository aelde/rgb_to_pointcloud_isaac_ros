#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
from PIL import Image as PILImage
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from std_msgs.msg import Header
import struct

class ImageToPointCloudPublisher(Node):

    def __init__(self):
        super().__init__('image_to_pointcloud_publisher')
        self.subscription = self.create_subscription(
            Image,
            '/tb_1_green/rgb',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, 'tb_1/pointcloud', 10)
        self.bridge = CvBridge()

        self.encoder = 'vitl'
        self.load_from = 'checkpoints/depth_anything_v2_metric_vkitti_vitl.pth'
        self.max_depth = 80 # hypersim=Indoor=20, kitti=outdoor=80
        
        #cal fol_x and fol_y
        horiz_aperture = 20.955
        vert_aperture = 15.2908
        focal_length = 12.3
        width = 1280
        height = 720
        
        
        self.focal_length_x = 751.3242797851562
        self.focal_length_y = 751.3242797851562
        print(f'f_x: {self.focal_length_x}, f_y: {self.focal_length_y}, width: {width}, height: {height}')
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        print(f'encoder: {self.encoder} , load_from: {self.load_from} , max_depth: {self.max_depth}')

        # Initialize the DepthAnythingV2 model with the specified configuration
        self.depth_anything = DepthAnythingV2(**{**model_configs[self.encoder], 'max_depth': self.max_depth})
        self.depth_anything.load_state_dict(torch.load(self.load_from, map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.DEVICE).eval()

        
    def listener_callback(self, msg):
        # self.get_logger().info('Receiving image')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Display the image
        # cv2.imshow("Received Image", cv_image)
        # cv2.waitKey(1)

        self.rgb_to_pointcloud(cv_image, msg.header)

    def rgb_to_pointcloud(self, cv_image, header):
        # Convert the cv_image to PIL image and get its size
        color_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        width, height = color_image.size
        # print(f'width: {width}, height: {height}')

        # Predict the depth
        pred = self.depth_anything.infer_image(cv_image, height)

        # Resize depth prediction to match the original image size
        resized_pred = PILImage.fromarray(pred).resize((width, height), PILImage.NEAREST)

        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / self.focal_length_x
        y = (y - height / 2) / self.focal_length_y
        z = np.array(resized_pred)
        
        # # default 
        # points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        points = np.stack((z, -(np.multiply(x, z)), -(np.multiply(y, z))), axis=-1).reshape(-1, 3)
        # points = np.stack((z, np.multiply(x, z), -(np.multiply(y, z))), axis=-1).reshape(-1, 3) # left is right, right is left
        colors = np.array(color_image).reshape(-1, 3)
        
        # points = np.stack((z, -x, -y), axis=-1).reshape(-1, 3)
        # colors = np.array(color_image).reshape(-1, 3)

        # Combine XYZ and RGB into a structured array
        cloud_data = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])

        cloud_data['x'] = points[:, 0]
        cloud_data['y'] = points[:, 1]
        cloud_data['z'] = points[:, 2]
        cloud_data['rgb'] = np.array((colors[:, 0] << 16) | (colors[:, 1] << 8) | (colors[:, 2]), dtype=np.uint32)
        
        header.frame_id = "odom"
        
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1),
        ]

        pc2 = point_cloud2.create_cloud(header, fields, cloud_data)

        self.publisher.publish(pc2)
        # self.get_logger().info('Published PointCloud2 message')

def main(args=None):
    rclpy.init(args=args)

    image_to_pointcloud_publisher = ImageToPointCloudPublisher()

    rclpy.spin(image_to_pointcloud_publisher)

    # Destroy the node explicitly
    image_to_pointcloud_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()