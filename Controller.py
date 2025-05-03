#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

INITIAL_POSITION = [0, 0]
EDGE_THRESHOLD = 0.1  # Threshold to detect proximity to an edge

class Triangle:
    def __init__(self, vertices, transition_direction=None):
        """
        :param vertices: List of 3 points representing the vertices of the triangle.
        :param transition_direction: Vector field direction towards the next triangle
        """
        self.vertices = np.array(vertices)  # Vertices of the triangle
        self.field_vectors = None  # To be set after the transition path is known
        self.transition_direction = transition_direction  # Direction towards next triangle
        self.next_triangle_idx = None  # To be set when transition logic is determined

    def set_field_vectors(self, field_vectors):
        """
        Set the vector field for the triangle.
        :param field_vectors: List of vectors corresponding to the vertices.
        """
        self.field_vectors = np.array(field_vectors)

    def set_next_triangle(self, next_triangle_idx):
        """
        Set the index of the next triangle to transition to.
        :param next_triangle_idx: The index of the next triangle in the path.
        """
        self.next_triangle_idx = next_triangle_idx

    def barycentric_coordinates(self, point):
        """
        Compute barycentric coordinates of the point with respect to the triangle.
        :param point: The point in the plane.
        :return: Barycentric coordinates [lambda1, lambda2, lambda3].
        """
        v0 = self.vertices[1] - self.vertices[0]
        v1 = self.vertices[2] - self.vertices[0]
        v2 = point - self.vertices[0]
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return np.array([1.0, 0.0, 0.0])
        lambda2 = (d11 * d20 - d01 * d21) / denom
        lambda3 = (d00 * d21 - d01 * d20) / denom
        lambda1 = 1.0 - lambda2 - lambda3
        return np.array([lambda1, lambda2, lambda3])

    def get_desired_velocity(self, point):
        """
        Calculate the desired velocity of the robot given the point inside the triangle.
        :param point: Current robot position.
        :return: The desired velocity (2D vector).
        """
        lambdas = self.barycentric_coordinates(point)
        return lambdas[0] * self.field_vectors[0] + lambdas[1] * self.field_vectors[1] + lambdas[2] * self.field_vectors[2]

    def is_point_near_edge(self, point, threshold=EDGE_THRESHOLD):
        """
        Check if the point (robot's position) is close to any of the triangle's edges.
        :param point: Current position of the robot.
        :param threshold: Distance threshold to consider proximity to edge.
        :return: True if the point is close to any edge of the triangle, False otherwise.
        """
        # Check distance to each edge of the triangle
        for i in range(3):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % 3]
            dist = self.distance_to_edge(point, p1, p2)
            if dist < threshold:
                return True
        return False

    def distance_to_edge(self, point, p1, p2):
        """
        Calculate the perpendicular distance from the point to the line segment defined by p1 and p2.
        :param point: Current position of the robot.
        :param p1, p2: The endpoints of the edge of the triangle.
        :return: Perpendicular distance to the edge.
        """
        # Vector from p1 to p2
        edge_vec = p2 - p1
        # Vector from p1 to the point
        point_vec = point - p1
        # Projection of point_vec onto edge_vec (dot product)
        edge_length = np.linalg.norm(edge_vec)
        if edge_length == 0:
            return np.linalg.norm(point - p1)  # If edge length is zero (degenerate case), return distance to p1
        projection = np.dot(point_vec, edge_vec) / edge_length
        # Find the closest point on the edge
        closest_point = p1 + projection * edge_vec / edge_length
        return np.linalg.norm(point - closest_point)

class TriangleGraph:
    def __init__(self, triangles):
        """
        :param triangles: List of Triangle objects.
        """
        self.triangles = triangles
        self.path = []  # Path for the transitions from one triangle to another

    def set_path(self, path):
        """
        Set the transition path between triangles.
        :param path: List of triangle indices representing the order of triangles.
        """
        self.path = path

    def get_next_triangle(self, current_triangle_idx):
        """
        Given the current triangle index, return the next triangle in the path.
        :param current_triangle_idx: Index of the current triangle.
        :return: Index of the next triangle.
        """
        return self.path[current_triangle_idx] if current_triangle_idx < len(self.path) else None

    def calculate_vector_fields(self):
        """
        Given the current path, calculate and set the vector field for each triangle in the path.
        The vector field should guide the robot towards the specific edge separating the triangles.
        """
        # Iterate over each triangle in the path (except the last one)
        for i in range(len(self.path) - 1):
            current_triangle = self.triangles[self.path[i]]
            next_triangle = self.triangles[self.path[i + 1]]

            # The transition edge is the one shared by the current triangle and the next triangle
            # Let's assume the transition edge is the edge that has a vertex in common between the two triangles
            transition_edge_idx = self.find_transition_edge(current_triangle, next_triangle)

            # Get the vertices of the transition edge
            p1 = current_triangle.vertices[transition_edge_idx]
            p2 = current_triangle.vertices[(transition_edge_idx + 1) % 3]  # The next vertex in the edge

            # Calculate the direction from p1 to p2 (transition edge)
            transition_direction = p2 - p1
            transition_direction /= np.linalg.norm(transition_direction)  # Normalize the direction

            # Set the vector field for the current triangle (towards the transition edge)
            current_triangle.set_field_vectors([transition_direction, transition_direction, transition_direction])

    def find_transition_edge(self, current_triangle, next_triangle):
        """
        Find the edge shared between two triangles (the edge that transitions from current to next triangle).
        :param current_triangle: The current triangle object.
        :param next_triangle: The next triangle object.
        :return: The index of the shared edge.
        """
        # Iterate through the edges of the current triangle and the next triangle
        for i in range(3):
            edge_current = set([tuple(current_triangle.vertices[i]), tuple(current_triangle.vertices[(i + 1) % 3])])
            for j in range(3):
                edge_next = set([tuple(next_triangle.vertices[j]), tuple(next_triangle.vertices[(j + 1) % 3])])
                if edge_current == edge_next:
                    return i  # Return the index of the edge from the current triangle
        return None  # If no shared edge is found (should not happen if path is correct)


class VectorFieldController(Node):
    def __init__(self, graph):
        super().__init__('vector_field_controller')

        # Publisher for velocity commands:
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribe to odometry for current robot position:
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.graph = graph  # Graph representing the triangulation and adjacency
        self.current_position = np.array(INITIAL_POSITION)
        self.current_triangle_idx = 0  # Start at the first triangle
        
        # Timer for the control loop
        timer_period = 0.1  # seconds (10 Hz)
        self.timer = self.create_timer(timer_period, self.control_loop_callback)

    def odom_callback(self, msg):
        """ Callback to update the robot's position from odometry """
        self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def transition_to_next_triangle(self):
        """
        Transition to the next triangle in the path, if robot is close to an edge.
        """
        current_triangle = self.graph.triangles[self.current_triangle_idx]
        
        # Check if the robot is close to any edge of the current triangle
        if current_triangle.is_point_near_edge(self.current_position, threshold=EDGE_THRESHOLD):
            next_triangle_idx = self.graph.get_next_triangle(self.current_triangle_idx)
            if next_triangle_idx is not None:
                self.current_triangle_idx = next_triangle_idx

    def run(self):
        """ Main control loop """
        while not rclpy.is_shutdown():
            # Transition to next triangle if near the edge
            self.transition_to_next_triangle()

            # Get the current triangle and compute the velocity
            current_triangle = self.graph.triangles[self.current_triangle_idx]
            desired_velocity = current_triangle.get_desired_velocity(self.current_position)

            # Create a Twist message to control the robot
            twist_msg = Twist()
            twist_msg.linear.x = desired_velocity[0]
            twist_msg.linear.y = desired_velocity[1]
            twist_msg.angular.z = 0.0  # No angular velocity for now

            # Publish the velocity command
            self.cmd_pub.publish(twist_msg)

    def control_loop_callback(self):
        """ Callback for the control loop timer """
        self.run()

def main(args=None):
    rclpy.init(args=args)

    # Example triangles with vertices
    triangles = [
        Triangle([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        Triangle([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]),
        Triangle([[1.0, 1.0], [2.0, 1.0], [2.0, 0.0]])
    ]

    # Define the path (transitioning between triangles)
    graph = TriangleGraph(triangles)
    graph.set_path([1, 2])  # Transition from triangle 0 to triangle 1, and then from 1 to 2

    # Calculate and set vector fields based on the path
    graph.calculate_vector_fields()

    # Create the controller and run the loop
    controller = VectorFieldController(graph)
    controller.run()

    rclpy.spin(controller)

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()