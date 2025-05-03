#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class TriangleVectorField:
    def __init__(self, vertices, field_vectors):
        """
        Initializes the vector field over a triangle.
        :param vertices: List of three vertices (each vertex is [x, y]).
        :param field_vectors: List of vector values at each vertex (each is [vx, vy]).
        """
        self.vertices = np.array(vertices)
        self.field_vectors = np.array(field_vectors)
    
    def barycentric_coordinates(self, point):
        """
        Compute barycentric coordinates of a point with respect to the triangle.
        :param point: The [x, y] coordinates of the point.
        :return: Array of barycentric coordinates [λ1, λ2, λ3].
        """
        # Let v0 = vertex2 - vertex1, v1 = vertex3 - vertex1, and v2 = point - vertex1.
        v0 = self.vertices[1] - self.vertices[0]
        v1 = self.vertices[2] - self.vertices[0]
        v2 = point - self.vertices[0]
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        # Avoid division by zero. This should not happen if triangle is non-degenerate.
        if denom == 0:
            return np.array([1.0, 0.0, 0.0])
        lambda2 = (d11 * d20 - d01 * d21) / denom
        lambda3 = (d00 * d21 - d01 * d20) / denom
        lambda1 = 1.0 - lambda2 - lambda3
        return np.array([lambda1, lambda2, lambda3])
    
    def get_desired_velocity(self, point):
        """
        Computes the desired velocity at a given point inside the triangle.
        :param point: Current position [x, y] of the robot.
        :return: Desired velocity vector [vx, vy] computed by affine interpolation.
        """
        lambdas = self.barycentric_coordinates(point)
        desired_vel = (lambdas[0] * self.field_vectors[0] +
                       lambdas[1] * self.field_vectors[1] +
                       lambdas[2] * self.field_vectors[2])
        return desired_vel

class VectorFieldController:
    def __init__(self):
        rospy.init_node('vector_field_controller', anonymous=True)
        # Publisher for velocity commands:
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # Subscribe to odometry for current robot position:
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Example: Define a triangle with vertices and assign vector values at the vertices.
        # In practice, these would be calculated based on your high-level plan and desired cell exit.
        vertices = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        # Example field_vectors: these determine the "flow" within the cell.
        # Here, they are chosen so that the desired flow pushes the robot toward one side (e.g., the exit boundary).
        field_vectors = [[0.5, 0.0], [0.5, 0.2], [0.5, -0.2]]
        self.triangle_field = TriangleVectorField(vertices, field_vectors)
        
        # Initialize robot position (assumed 2D):
        self.current_position = np.array([0.0, 0.0])
        self.rate = rospy.Rate(10)  # 10 Hz loop rate
    
    def odom_callback(self, msg):
        """
        Callback to update the current robot position from odometry.
        """
        self.current_position = np.array([msg.pose.pose.position.x,
                                           msg.pose.pose.position.y])
    
    def run(self):
        """
        Main control loop that computes and publishes velocity commands.
        """
        while not rospy.is_shutdown():
            # Compute the desired velocity from the vector field for the current position.
            desired_vel = self.triangle_field.get_desired_velocity(self.current_position)
            
            # Create and publish a Twist message.
            twist_msg = Twist()
            twist_msg.linear.x = desired_vel[0]
            twist_msg.linear.y = desired_vel[1]
            # For simplicity, we set angular velocity to zero.
            twist_msg.angular.z = 0.0
            
            self.cmd_pub.publish(twist_msg)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = VectorFieldController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
