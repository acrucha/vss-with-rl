from rsoccer_gym.Render import VSSRobot

TARGET           = (255, 50 , 50 )
TARGET_TAG_BLUE  = (138, 164, 255)
TARGET_TAG_GREEN = (204, 220, 153)

class Target(VSSRobot):
    def __init__(self, x, y, angle, scale):
        super().__init__(x, y, angle, scale, 0, team_color=TARGET_TAG_BLUE)
        self.id_color = TARGET_TAG_GREEN