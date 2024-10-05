from collections import namedtuple
from dataclasses import dataclass
import math
from rsoccer_gym.Entities import Robot

@dataclass()
class RobotWithPID(Robot):
    kP: float = 0.0
    kI: float = 0.0
    kD: float = 0.0

Point2D = namedtuple("Point2D", ["x", "y"])

def dist_to(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the distance between two points"""
    return ((p_1.x - p_2.x) ** 2 + (p_1.y - p_2.y) ** 2) ** 0.5


def math_modularize(value: float, mod: float) -> float:
    """Make a value modular between 0 and mod"""
    if not -mod <= value <= mod:
        value = math.fmod(value, mod)

    if value < 0:
        value += mod

    return value

def smallest_angle_diff(angle_a: float, angle_b: float) -> float:
    """Returns the smallest angle difference between two angles"""
    angle: float = math_modularize(angle_b - angle_a, 2 * math.pi)
    if angle >= math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle

def abs_smallest_angle_diff(angle_a: float, angle_b: float) -> float:
    """Returns the absolute smallest angle difference between two angles"""
    return abs(smallest_angle_diff(angle_a, angle_b))
