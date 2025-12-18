import os
import time
import sys
sys.path.insert(0, "/workspace/inao")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)
from inao import NAO
from naoqi import ALProxy

class NaoSays:
    def __init__(self, ip, port):
        self.tts = ALProxy("ALTextToSpeech", ip, port)
        self.leds = ALProxy("ALLeds", ip, port)
        
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
        ttsProxy = ALProxy("ALTextToSpeech", self.robotIP, self.PORT)
        postureProxy = ALProxy("ALRobotPosture",self.robotIP, self.PORT)
        motionProxy = ALProxy("ALMotion",self.robotIP, self.PORT)
        postureProxy.goToPosture(posture_name, speed)
        ttsProxy.say("Movement {} complete!".format(posture_name))
        motionProxy.rest()

    def move_position(self, x=0.0, y=0.0, theta=0.0):
        """
        Move the robot
        :param x: forward/backward movement in meters
        :param y: left/right movement in meters
        :param theta: rotation in radians
        :return: None
        """
        ttsProxy = ALProxy("ALTextToSpeech", self.robotIP, self.PORT)
        motionProxy = ALProxy("ALMotion",self.robotIP, self.PORT)
        motionProxy.wakeUp()
        motionProxy.moveTo(x, y, theta)
        ttsProxy.say("Movement complete!")
        motionProxy.rest()

    def move_joint(self, joint_name="HeadYaw", angle=0.0, speed=0.1, waitingtime=2.0):
        """
        Move a specific joint
        :param joint_name: name of the joint
        :param angle: angle in radians
        :param speed: speed of the movement (0.0 to 1.0)
        :return: None
        """
        ttsProxy = ALProxy("ALTextToSpeech", self.robotIP, self.PORT)
        motionProxy = ALProxy("ALMotion",self.robotIP, self.PORT)
        motionProxy.wakeUp()
        motionProxy.setStiffnesses(joint_name, 1.0)
        motionProxy.setAngles(joint_name, angle, speed)
        ttsProxy.say("Movement of joint {} complete!".format(joint_name))
        time.sleep(waitingtime)
        motionProxy.setStiffnesses(joint_name, 0.0)
        motionProxy.rest()

if __name__ == "__main__":
    ROBOT_IP = "192.168.1.118"
    PORT = 9559
    robot = NaoSays(ROBOT_IP, PORT)
    robot.change_eye_color("yellow")
    # game.record_audio(5)