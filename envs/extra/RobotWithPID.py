from dataclasses import dataclass
from rsoccer_gym.Entities import Robot

@dataclass()
class RobotWithPID(Robot):
    kP: float = 0.0
    kI: float = 0.0
    kD: float = 0.0
