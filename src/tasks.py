import time
import io
from PIL import Image

from inao import NAO
from naoqi import ALProxy

class NaoTaskExecutor:
    def __init__(self, ip, port):
        self.tts_proxy = ALProxy("ALTextToSpeech", ip, port)
        self.motion_proxy = ALProxy("ALMotion", ip, port)
        self.posture_proxy = ALProxy("ALRobotPosture", ip, port)
        self.leds_proxy = ALProxy("ALLeds", ip, port)
        self.cam_proxy = ALProxy("ALVideoDevice", ip, port)
        self.resolution = 2  # VGA
        self.colorSpace = 11  # RGB

    def close(self):
        if self.video_client:
            self.cam_proxy.unsubscribe(self.video_client)
            self.video_client = None
    
    # lead reaction
    def change_eye_color(self, color):
        color_map = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0)
            }
        if color in color_map:
            rgb = color_map[color]
            self.leds.fadeRGB("FaceLeds", rgb[0], rgb[1], rgb[2], 0.5)
        time.sleep(1)
        self.leds.fadeRGB("FaceLeds", 1.0, 1.0, 1.0, 0.3)
        time.sleep(0.3)

    # motion reaction
    def posture(self, posture_name="stand", speed=1.0):
        """
        Set the robot in a given posture
        :param posture_name: name of the posture
        :param speed: speed of the movement
        :return: None
        """ 
        self.posture_proxy.goToPosture(posture_name, speed)
        self.tts_proxy.say("Movement {} complete!".format(posture_name))
        self.motion_proxy.rest()

    def move_position(self, x=0.0, y=0.0, theta=0.0):
        """
        Move the robot
        :param x: forward/backward movement in meters
        :param y: left/right movement in meters
        :param theta: rotation in radians
        :return: None
        """
        self.motion_proxy.wakeUp()
        self.motion_proxy.moveTo(x, y, theta)
        self.tts_proxy.say("Movement complete!")
        self.motion_proxy.rest()

    def move_joint(self, joint_name="HeadYaw", angle=0.0, speed=0.1, waitingtime=2.0):
        """
        Move a specific joint
        :param joint_name: name of the joint
        :param angle: angle in radians
        :param speed: speed of the movement (0.0 to 1.0)
        :return: None
        """
        self.motion_proxy.wakeUp()
        self.motion_proxy.setStiffnesses(joint_name, 1.0)
        self.motion_proxy.setAngles(joint_name, angle, speed)
        self.tts_proxy.say("Movement of joint {} complete!".format(joint_name))
        time.sleep(waitingtime)
        self.motion_proxy.setStiffnesses(joint_name, 0.0)
        self.motion_proxy.rest()
    
    # vision
    def capture_frame(self):
        image = self.cam_proxy.getImageRemote(self.video_client)
        width = image[0]
        height = image[1]
        rgb_bytes = image[6]

        img = Image.frombytes("RGB", (width, height), rgb_bytes)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()